"""Conversation Engine orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

from src.models import (
    ArticleReference,
    ChatResponse,
    Citation,
    ConversationMessage,
    ConversationState,
    MessageRole,
    ProactiveAnalysis,
    ResolutionStatus,
    RetrievalResult,
    Severity,
)

from .context import (
    ClarificationPlanner,
    ContextUpdater,
    RetrievalQueryBuilder,
    reset_for_new_request,
)
from .intent import ConversationIntent, IntentDetector
from .interfaces import (
    ConversationMemory,
    GeneratedResponse,
    GenerationContext,
    LLMService,
    ProactiveService,
    Retriever,
)
from .memory import InMemoryConversationMemory
from .troubleshooting import TroubleshootingTracker

logger = logging.getLogger(__name__)


SYSTEM_INSTRUCTIONS = """You are a customer-support response generator.
Use only the retrieved knowledge as factual support. Clearly state uncertainty.
Never invent citations or unsupported troubleshooting steps. Do not repeat any
excluded failed step. Give concise, actionable guidance and ask the customer to
confirm whether the guidance resolved the issue.
""".strip()


@dataclass(frozen=True)
class ConversationEngineOptions:
    """Transparent orchestration thresholds and context bounds."""

    top_k: int = 5
    recent_message_limit: int = 6
    medium_confidence: float = 0.45
    high_confidence: float = 0.75
    high_frustration: float = 0.8
    failed_attempts_before_escalation: int = 2

    def __post_init__(self) -> None:
        if self.top_k < 1 or self.recent_message_limit < 0:
            raise ValueError("top_k must be positive and recent_message_limit non-negative")
        if not 0 <= self.medium_confidence <= self.high_confidence <= 1:
            raise ValueError("confidence thresholds must satisfy 0 <= medium <= high <= 1")
        if not 0 <= self.high_frustration <= 1:
            raise ValueError("high_frustration must be between 0 and 1")
        if self.failed_attempts_before_escalation < 1:
            raise ValueError("failed_attempts_before_escalation must be positive")


@dataclass(frozen=True)
class EscalationDecision:
    required: bool
    reason: Optional[str] = None


class ConversationEngine:
    """Coordinate state, proactive signals, retrieval, and grounded generation."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        llm_service: LLMService,
        proactive_service: Optional[ProactiveService] = None,
        memory: Optional[ConversationMemory] = None,
        options: Optional[ConversationEngineOptions] = None,
    ) -> None:
        self._retriever = retriever
        self._llm_service = llm_service
        self._proactive_service = proactive_service
        self._memory = memory or InMemoryConversationMemory()
        self._options = options or ConversationEngineOptions()
        self._intent_detector = IntentDetector()
        self._context_updater = ContextUpdater()
        self._clarification_planner = ClarificationPlanner()
        self._query_builder = RetrievalQueryBuilder()
        self._troubleshooting = TroubleshootingTracker()

    def handle_message(
        self,
        *,
        conversation_id: str,
        user_message: str,
        user_id: Optional[str] = None,
    ) -> ChatResponse:
        """Handle one user turn and persist its resulting structured state."""

        conversation_id = conversation_id.strip()
        user_message = " ".join(user_message.strip().split())
        if not conversation_id:
            raise ValueError("conversation_id must not be empty")
        if not user_message:
            raise ValueError("user_message must not be empty")

        state = self._memory.load(conversation_id) or ConversationState(
            conversation_id=conversation_id,
            user_id=user_id,
        )
        if state.user_id and user_id and state.user_id != user_id:
            raise ValueError("conversation_id is already associated with another user")
        if state.user_id is None and user_id:
            state.user_id = user_id

        intent = self._intent_detector.detect(user_message, state)
        if intent is ConversationIntent.NEW_REQUEST:
            state = reset_for_new_request(state)

        state = self._context_updater.update(state, user_message, intent)
        state.messages.append(
            ConversationMessage(role=MessageRole.USER, content=user_message)
        )
        state.turn_count += 1

        if intent is ConversationIntent.TROUBLESHOOTING_RESULT:
            state = self._troubleshooting.record_failed_attempt(state, user_message)

        if intent is ConversationIntent.RESOLUTION_CONFIRMATION:
            state.resolution_status = ResolutionStatus.RESOLVED
            return self._finish(
                state,
                ChatResponse(
                    message="Great — I've marked this issue as resolved.",
                    confidence=1.0,
                ),
            )

        proactive = self._analyze_proactive(user_message, state)
        decision = self._escalation_decision(intent, state, proactive)
        if decision.required:
            state.resolution_status = ResolutionStatus.ESCALATED
            return self._finish(
                state,
                ChatResponse(
                    message=self._escalation_message(decision.reason),
                    escalation_required=True,
                    related_articles=proactive.recommended_articles,
                ),
            )

        clarification = self._clarification_planner.next_question(state)
        if clarification is not None:
            state.resolution_status = ResolutionStatus.NEEDS_CLARIFICATION
            return self._finish(
                state,
                ChatResponse(
                    message=clarification.question,
                    related_articles=proactive.recommended_articles,
                ),
            )

        state.resolution_status = ResolutionStatus.READY_TO_RESOLVE
        query, filters = self._query_builder.build(state, user_message)
        try:
            results = list(
                self._retriever.search(
                    query=query,
                    filters=filters,
                    top_k=self._options.top_k,
                )
            )
        except Exception:
            logger.exception(
                "Conversation retrieval failed",
                extra={"conversation_id": conversation_id},
            )
            return self._knowledge_fallback(
                state,
                proactive.recommended_articles,
                "The support knowledge service is temporarily unavailable.",
            )

        retrieval_confidence = self._retrieval_confidence(results)
        if not results:
            return self._knowledge_fallback(
                state,
                proactive.recommended_articles,
                "I couldn't find sufficient supported knowledge for this issue.",
            )
        if retrieval_confidence < self._options.medium_confidence:
            return self._knowledge_fallback(
                state,
                proactive.recommended_articles,
                "The available knowledge has low confidence for this issue.",
                confidence=retrieval_confidence,
            )

        generation_context = self._generation_context(
            state=state,
            current_user_message=user_message,
            proactive=proactive,
            results=results,
            cautious=retrieval_confidence < self._options.high_confidence,
        )
        try:
            generated = self._llm_service.generate(generation_context)
            if not generated.message.strip():
                raise ValueError("LLM service returned an empty message")
        except Exception:
            logger.exception(
                "Conversation response generation failed",
                extra={"conversation_id": conversation_id},
            )
            state.resolution_status = ResolutionStatus.ESCALATED
            return self._finish(
                state,
                ChatResponse(
                    message=(
                        "I found relevant support knowledge but couldn't generate a safe "
                        "response. A human support agent should review the issue."
                    ),
                    citations=self._citations(results),
                    escalation_required=True,
                    confidence=retrieval_confidence,
                    related_articles=proactive.recommended_articles,
                ),
            )

        state, actions = self._troubleshooting.add_suggestions(
            state, generated.suggested_actions
        )
        if generated.suggested_actions and not actions and state.attempted_steps:
            state.resolution_status = ResolutionStatus.ESCALATED
            return self._finish(
                state,
                ChatResponse(
                    message=(
                        "The generated guidance only repeated steps you already tried. "
                        "A human support agent should review the issue."
                    ),
                    citations=self._citations(results),
                    escalation_required=True,
                    confidence=self._combined_confidence(
                        retrieval_confidence, generated
                    ),
                    related_articles=proactive.recommended_articles,
                ),
            )

        state.resolution_status = ResolutionStatus.AWAITING_CONFIRMATION
        return self._finish(
            state,
            ChatResponse(
                message=generated.message.strip(),
                citations=self._citations(results),
                suggested_actions=actions,
                confidence=self._combined_confidence(retrieval_confidence, generated),
                related_articles=proactive.recommended_articles,
            ),
        )

    def get_state(self, conversation_id: str) -> Optional[ConversationState]:
        """Return a defensive copy of current state for integration consumers."""

        return self._memory.load(conversation_id)

    def _analyze_proactive(
        self, message: str, state: ConversationState
    ) -> ProactiveAnalysis:
        if self._proactive_service is None:
            return ProactiveAnalysis()
        try:
            return self._proactive_service.analyze(
                message=message,
                conversation=state.model_copy(deep=True),
            )
        except Exception:
            logger.exception(
                "Proactive analysis failed",
                extra={"conversation_id": state.conversation_id},
            )
            return ProactiveAnalysis()

    def _escalation_decision(
        self,
        intent: ConversationIntent,
        state: ConversationState,
        proactive: ProactiveAnalysis,
    ) -> EscalationDecision:
        if intent is ConversationIntent.ESCALATION_REQUEST:
            return EscalationDecision(True, "explicit_human_request")
        if proactive.escalation_required:
            return EscalationDecision(True, proactive.reason or "proactive_signal")
        if proactive.frustration_score >= self._options.high_frustration:
            return EscalationDecision(True, "high_frustration")
        if state.severity is Severity.CRITICAL:
            return EscalationDecision(True, "critical_severity")
        if state.severity is Severity.HIGH and state.attempted_steps:
            return EscalationDecision(True, "high_severity_failed_attempt")
        if len(state.attempted_steps) >= self._options.failed_attempts_before_escalation:
            return EscalationDecision(True, "repeated_failed_troubleshooting")
        return EscalationDecision(False)

    def _generation_context(
        self,
        *,
        state: ConversationState,
        current_user_message: str,
        proactive: ProactiveAnalysis,
        results: Sequence[RetrievalResult],
        cautious: bool,
    ) -> GenerationContext:
        previous_messages = state.messages[:-1]
        limit = self._options.recent_message_limit
        recent = previous_messages[-limit:] if limit else []
        structured_state = state.model_copy(deep=True)
        structured_state.messages = []
        return GenerationContext(
            system_instructions=SYSTEM_INSTRUCTIONS,
            conversation=structured_state,
            recent_messages=tuple(message.model_copy(deep=True) for message in recent),
            conversation_summary=self._memory.get_summary(state.conversation_id),
            retrieved_knowledge=tuple(
                result.model_copy(deep=True) for result in results
            ),
            proactive_analysis=proactive,
            current_user_message=current_user_message,
            excluded_steps=tuple(state.attempted_steps),
            cautious=cautious,
        )

    def _knowledge_fallback(
        self,
        state: ConversationState,
        related_articles: list[ArticleReference],
        explanation: str,
        confidence: Optional[float] = None,
    ) -> ChatResponse:
        state.resolution_status = ResolutionStatus.ESCALATED
        return self._finish(
            state,
            ChatResponse(
                message=(
                    f"{explanation} I won't guess at troubleshooting steps; "
                    "a human support agent should review the issue."
                ),
                escalation_required=True,
                confidence=confidence,
                related_articles=related_articles,
            ),
        )

    def _finish(self, state: ConversationState, response: ChatResponse) -> ChatResponse:
        state.messages.append(
            ConversationMessage(role=MessageRole.ASSISTANT, content=response.message)
        )
        self._memory.save(state)
        return response

    @staticmethod
    def _retrieval_confidence(results: Sequence[RetrievalResult]) -> float:
        if not results:
            return 0.0
        return max(0.0, min(1.0, max(result.score for result in results)))

    @staticmethod
    def _combined_confidence(
        retrieval_confidence: float, generated: GeneratedResponse
    ) -> float:
        if generated.confidence is None:
            return retrieval_confidence
        generation_confidence = max(0.0, min(1.0, generated.confidence))
        return min(retrieval_confidence, generation_confidence)

    @staticmethod
    def _citations(results: Sequence[RetrievalResult]) -> list[Citation]:
        citations: list[Citation] = []
        for result in results:
            source = result.metadata.get("source") or result.metadata.get("title")
            if not isinstance(source, str) or not source.strip():
                source = result.document_id
            citations.append(
                Citation(source=source, document_id=result.document_id)
            )
        return citations

    @staticmethod
    def _escalation_message(reason: Optional[str]) -> str:
        messages = {
            "explicit_human_request": "I'll escalate this conversation to a human support agent.",
            "high_frustration": (
                "I can see this has been frustrating. I'll escalate it to a human "
                "support agent so you don't have to repeat failed steps."
            ),
            "critical_severity": (
                "This appears critical, so a human support agent should take over."
            ),
            "high_severity_failed_attempt": (
                "A high-severity issue is still unresolved after troubleshooting, so "
                "a human support agent should take over."
            ),
            "repeated_failed_troubleshooting": (
                "Multiple supported troubleshooting steps have failed, so a human "
                "support agent should take over."
            ),
        }
        return messages.get(
            reason,
            "The proactive support signal recommends escalation to a human support agent.",
        )
