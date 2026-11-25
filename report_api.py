"""
report_api.py — REST API Endpoints for Intelligent Report Generation in EvoRAG
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlmodel import Session, select

from auth import get_current_user, get_optional_current_user
from database import get_session
from models import (
    AnalysisSnapshot,
    GeneratedReport,
    ReportGenerateRequest,
    ReportRead,
    User,
)
from report_service import (
    generate_report_from_snapshot,
    get_or_create_snapshot_for_message,
    get_snapshot_by_id,
)

router = APIRouter(prefix="/reports", tags=["reports"])


@router.post("/generate", response_model=ReportRead, status_code=status.HTTP_201_CREATED)
def generate_report_endpoint(
    req: ReportGenerateRequest,
    current_user: Optional[User] = Depends(get_optional_current_user),
    session: Session = Depends(get_session),
):
    """
    Generate an Executive Summary or Detailed Multi-Persona Report (PDF or DOCX).
    Reuses existing AnalysisSnapshot — zero redundant RAG or persona inferences.
    """
    snapshot: Optional[AnalysisSnapshot] = None

    if req.snapshot_id:
        snapshot = get_snapshot_by_id(session, req.snapshot_id)
    elif req.message_id or req.conversation_id:
        snapshot = get_or_create_snapshot_for_message(
            session=session,
            message_id=req.message_id or "",
            conversation_id=req.conversation_id,
            user_id=current_user.id if current_user else None,
        )

    if not snapshot:
        raise HTTPException(
            status_code=404,
            detail="Analysis snapshot not found. Please provide a valid snapshot_id or message_id."
        )

    try:
        chosen_format = req.file_format or req.format or "pdf"
        report = generate_report_from_snapshot(
            session=session,
            snapshot_id=snapshot.snapshot_id,
            report_type=req.report_type,
            fmt=chosen_format,
            requesting_user=current_user,
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")

    return ReportRead(
        id=report.id,
        report_id=report.id,
        snapshot_id=report.snapshot_id,
        user_id=report.user_id,
        conversation_id=report.conversation_id,
        report_type=report.report_type,
        format=report.format,
        title=report.title,
        file_size_kb=report.file_size_kb,
        download_url=f"/reports/{report.id}/download",
        created_at=report.created_at.isoformat(),
    )


@router.get("/{report_id}/download", summary="Download generated PDF or DOCX report")
def download_report_endpoint(
    report_id: str,
    current_user: Optional[User] = Depends(get_optional_current_user),
    session: Session = Depends(get_session),
):
    """Securely download a generated PDF or DOCX report with path traversal safeguards."""
    report = session.get(GeneratedReport, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found.")

    if report.user_id and current_user and report.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied to this report.")

    path = Path(report.file_path).resolve()
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Report file not found on disk.")

    media_type = (
        "application/pdf"
        if report.format.lower() == "pdf"
        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

    filename = path.name
    return FileResponse(
        path=str(path),
        media_type=media_type,
        filename=filename,
    )


@router.get("/{report_id}/preview", summary="Preview structured content of a report")
def preview_report_endpoint(
    report_id: str,
    current_user: Optional[User] = Depends(get_optional_current_user),
    session: Session = Depends(get_session),
):
    """Retrieve structured JSON preview of report contents without downloading binary file."""
    import json
    report = session.get(GeneratedReport, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found.")
    snapshot = get_snapshot_by_id(session, report.snapshot_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Underlying analysis snapshot missing.")

    return {
        "report_id": report.id,
        "title": report.title,
        "report_type": report.report_type,
        "format": report.format,
        "query": snapshot.query,
        "domain": snapshot.domain,
        "executive_summary": snapshot.final_answer,
        "top_personas": json.loads(snapshot.top_personas_json or "[]"),
        "sources": json.loads(snapshot.sources_json or "[]")[:6],
        "created_at": report.created_at.isoformat(),
        "download_url": f"/reports/{report.id}/download",
    }


@router.get("/history", response_model=List[ReportRead], summary="List previously generated reports")
def list_reports_history(
    conversation_id: Optional[int] = Query(None),
    current_user: Optional[User] = Depends(get_optional_current_user),
    session: Session = Depends(get_session),
):
    """List reports generated by the user or within a specific conversation."""
    stmt = select(GeneratedReport)
    if current_user:
        stmt = stmt.where(GeneratedReport.user_id == current_user.id)
    elif conversation_id:
        stmt = stmt.where(GeneratedReport.conversation_id == conversation_id)
    else:
        # Return recent reports
        stmt = stmt.limit(20)

    stmt = stmt.order_by(GeneratedReport.created_at.desc())
    reports = session.exec(stmt).all()

    return [
        ReportRead(
            id=r.id,
            snapshot_id=r.snapshot_id,
            user_id=r.user_id,
            conversation_id=r.conversation_id,
            report_type=r.report_type,
            format=r.format,
            title=r.title,
            file_size_kb=r.file_size_kb,
            download_url=f"/reports/{r.id}/download",
            created_at=r.created_at.isoformat(),
        )
        for r in reports
    ]
