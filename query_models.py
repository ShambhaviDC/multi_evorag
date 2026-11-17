"""query_models.py - Pydantic Schemas for Structured Query Representation and Adaptive Outputs (EvoRAG Phase 7)."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EntityItem(BaseModel):
    name: str = Field(..., description="Entity name e.g. NVIDIA, Quantum Computing, US")
    type: str = Field(..., description="Person, Organization, Company, Technology, Event, Location, Date/Time, Concept, Metric, Product")
    attributes: Optional[Dict[str, Any]] = Field(default=None, description="Optional entity attributes")


class RelationshipItem(BaseModel):
    source: str = Field(..., description="Source entity")
    target: str = Field(..., description="Target entity")
    type: str = Field(..., description="competes_with, acquired, works_for, located_in, uses, part_of, occurred_at, related_to")


class QueryConstraints(BaseModel):
    requires_latest_information: bool = Field(default=False)
    requires_source_citations: bool = Field(default=True)
    temporal_focus: Optional[str] = Field(default=None, description="e.g. 2026, latest, historical, Q3")


class OutputPlan(BaseModel):
    primary_format: str = Field(
        default="text_summary",
        description="text_summary, comparison_table, comparison_chart, bar_chart, line_chart, pie_chart, timeline, numerical_summary, image_analysis, relationship_graph"
    )
    additional_formats: List[str] = Field(
        default_factory=list,
        description="List of secondary format components e.g. ['comparison_table', 'source_list']"
    )


class StructuredQuery(BaseModel):
    original_query: str = Field(..., description="Preserved user question")
    intent: str = Field(
        default="factual_question_answering",
        description="factual_question_answering, news_research, latest_news_retrieval, comparative_analysis, summarization, explanation, trend_analysis, numerical_analysis, document_qa, image_understanding, timeline_generation, entity_relationship_analysis"
    )
    query_category: str = Field(
        default="factual",
        description="factual, news, comparative, numerical, document, image, historical, relational"
    )
    entities: List[EntityItem] = Field(default_factory=list)
    relationships: List[RelationshipItem] = Field(default_factory=list)
    requested_information: List[str] = Field(default_factory=list)
    constraints: QueryConstraints = Field(default_factory=QueryConstraints)
    output_plan: OutputPlan = Field(default_factory=OutputPlan)
    
    # Internal routing flags computed by Input Understanding Agent
    requires_news_retrieval: bool = Field(default=False)
    requires_document_analysis: bool = Field(default=False)
    requires_image_analysis: bool = Field(default=False)
    requires_numerical_analysis: bool = Field(default=False)
    attachment_filenames: List[str] = Field(default_factory=list)


class UIComponent(BaseModel):
    type: str = Field(..., description="comparison_table, bar_chart, line_chart, pie_chart, timeline, numerical_summary, relationship_graph, file_summary, source_list")
    title: Optional[str] = Field(default=None)
    data: Dict[str, Any] = Field(default_factory=dict)


class AdaptiveConsensusResponse(BaseModel):
    final_answer: str
    components: List[UIComponent] = Field(default_factory=list)
    structured_query: Optional[StructuredQuery] = None
    top_personas: List[str] = Field(default_factory=list)
    scores: Dict[str, float] = Field(default_factory=dict)
    all_responses: Dict[str, Optional[str]] = Field(default_factory=dict)
    score_matrix: Dict[str, Dict[str, Optional[Dict[str, float]]]] = Field(default_factory=dict)
    query_id: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    attachments_processed: List[Dict[str, Any]] = Field(default_factory=list)
