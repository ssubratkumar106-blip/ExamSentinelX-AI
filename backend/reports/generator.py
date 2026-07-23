"""
backend/reports/generator.py — PDF Report Generator
=====================================================
PURPOSE: Generate professional PDF exam proctoring reports.

WHAT IS INCLUDED IN THE REPORT:
    1. Student information
    2. Exam details (title, duration, score)
    3. Session summary (start/end time, risk score)
    4. Violation timeline with type, confidence, timestamp
    5. Statistical summary (bar chart of violation types)
    6. Admin signature area

LIBRARY: fpdf2 (modern version of FPDF for Python)
    - Generates PDF from scratch with full layout control
    - No external tools needed (pure Python)
    - Supports Unicode, tables, images
"""

import os
from datetime import datetime
from fpdf import FPDF
from database.models import ExamSession, ViolationLog

import logging
logger = logging.getLogger(__name__)


class ExamReport(FPDF):
    """
    Custom FPDF subclass for ExamGuard reports.
    Overrides header() and footer() for consistent branding.
    """

    def header(self):
        """Printed at top of every page."""
        # Brand bar
        self.set_fill_color(15, 23, 42)   # Dark navy
        self.rect(0, 0, 210, 18, 'F')

        self.set_text_color(99, 179, 237)  # Blue accent
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 18, '  ExamGuard AI — Proctoring Report', ln=True, align='L')

        self.set_text_color(0, 0, 0)
        self.ln(5)

    def footer(self):
        """Printed at bottom of every page."""
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10,
                  f'ExamGuard AI — Confidential Report | Page {self.page_no()}/{{nb}} | '
                  f'Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}',
                  align='C')

    def colored_cell(self, w, h, txt, color_rgb, text_color=(255, 255, 255), border=0, align='C', fill=True):
        """Utility: cell with custom background and text color."""
        self.set_fill_color(*color_rgb)
        self.set_text_color(*text_color)
        self.cell(w, h, txt, border=border, align=align, fill=fill)
        self.set_text_color(0, 0, 0)  # Reset


def generate_session_report(session: ExamSession) -> str:
    """
    Generate a PDF report for an exam session.
    
    Args:
        session: ExamSession ORM object
        
    Returns:
        File path of the generated PDF, or empty string on failure.
    """
    try:
        output_dir = 'reports/generated'
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f'report_session_{session.id}_{timestamp}.pdf'
        output_path = os.path.join(output_dir, filename)

        pdf = ExamReport()
        pdf.alias_nb_pages()  # Enable {nb} in footer
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=20)

        # ── SECTION 1: Report Title ────────────────────────────────────────────
        pdf.set_font('Helvetica', 'B', 20)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 12, 'PROCTORING SESSION REPORT', ln=True, align='C')
        pdf.ln(2)

        # Divider line
        pdf.set_draw_color(99, 179, 237)
        pdf.set_line_width(0.8)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(8)

        # ── SECTION 2: Student & Exam Info ────────────────────────────────────
        pdf.set_font('Helvetica', 'B', 12)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 8, 'STUDENT & EXAM INFORMATION', ln=True)
        pdf.ln(2)

        info_rows = [
            ('Student Name', session.student.full_name),
            ('Student ID', f'#{session.student.id:04d}'),
            ('Username', session.student.username),
            ('Email', session.student.email),
            ('Exam Title', session.exam.title),
            ('Subject', session.exam.subject or 'General'),
            ('Session ID', f'#{session.id:06d}'),
            ('IP Address', session.ip_address or 'N/A'),
        ]

        pdf.set_font('Helvetica', '', 10)
        col_w = 45
        for label, value in info_rows:
            pdf.set_fill_color(241, 245, 249)
            pdf.cell(col_w, 8, f'  {label}:', border=1, fill=True)
            pdf.set_fill_color(255, 255, 255)
            pdf.cell(145, 8, f'  {value}', border=1, fill=True)
            pdf.ln()

        pdf.ln(8)

        # ── SECTION 3: Session Timeline ───────────────────────────────────────
        pdf.set_font('Helvetica', 'B', 12)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 8, 'SESSION TIMELINE', ln=True)
        pdf.ln(2)

        pdf.set_font('Helvetica', '', 10)
        timeline_rows = [
            ('Start Time', session.started_at.strftime('%Y-%m-%d %H:%M:%S UTC')),
            ('End Time', session.ended_at.strftime('%Y-%m-%d %H:%M:%S UTC') if session.ended_at else 'In Progress'),
            ('Duration', session.duration_display),
            ('Status', session.status.upper()),
            ('Score', f'{session.score}/{session.exam.total_marks}' if session.score is not None else 'N/A'),
        ]

        for label, value in timeline_rows:
            pdf.set_fill_color(241, 245, 249)
            pdf.cell(col_w, 8, f'  {label}:', border=1, fill=True)
            status_color = {'FLAGGED': (254, 226, 226), 'COMPLETED': (220, 252, 231)}.get(value, (255, 255, 255))
            pdf.set_fill_color(*status_color)
            pdf.cell(145, 8, f'  {value}', border=1, fill=True)
            pdf.ln()

        pdf.ln(8)

        # ── SECTION 4: Risk Assessment ────────────────────────────────────────
        pdf.set_font('Helvetica', 'B', 12)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 8, 'RISK ASSESSMENT', ln=True)
        pdf.ln(2)

        risk = session.risk_score
        risk_color = (254, 226, 226) if risk >= 7 else (254, 243, 199) if risk >= 4 else (220, 252, 231)
        risk_label = 'HIGH RISK' if risk >= 7 else 'MEDIUM RISK' if risk >= 4 else 'LOW RISK'

        pdf.set_font('Helvetica', 'B', 16)
        pdf.set_fill_color(*risk_color)
        pdf.cell(0, 14, f'Risk Score: {risk:.1f}/10.0  ({risk_label})', border=1, align='C', fill=True, ln=True)
        pdf.ln(2)

        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(80, 80, 80)
        pdf.multi_cell(0, 6, f'Total Violations: {session.total_violations}  |  '
                              f'Exam Integrity: {"COMPROMISED" if risk >= 7 else "QUESTIONABLE" if risk >= 4 else "MAINTAINED"}')
        pdf.ln(6)

        # ── SECTION 5: Violation Log Table ────────────────────────────────────
        violations = session.violations.order_by(ViolationLog.timestamp.asc()).all()

        pdf.set_font('Helvetica', 'B', 12)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 8, f'VIOLATION LOG ({len(violations)} events)', ln=True)
        pdf.ln(2)

        if violations:
            # Table header
            pdf.set_font('Helvetica', 'B', 9)
            headers = [('#', 12), ('Timestamp', 48), ('Violation Type', 60), ('Confidence', 30), ('Severity', 30)]
            header_bg = (15, 23, 42)

            for header, width in headers:
                pdf.colored_cell(width, 8, header, header_bg)
            pdf.ln()

            # Table rows
            pdf.set_font('Helvetica', '', 9)
            severity_colors = {'High': (254, 226, 226), 'Medium': (254, 243, 199), 'Low': (220, 252, 231)}

            for i, v in enumerate(violations):
                row_fill = (248, 250, 252) if i % 2 == 0 else (255, 255, 255)
                pdf.set_fill_color(*row_fill)
                pdf.set_text_color(0, 0, 0)

                pdf.cell(12, 7, str(i + 1), border=1, align='C', fill=True)
                pdf.cell(48, 7, v.timestamp.strftime('%H:%M:%S'), border=1, fill=True)
                pdf.cell(60, 7, v.violation_display, border=1, fill=True)
                pdf.cell(30, 7, f'{v.confidence:.0%}' if v.confidence else 'N/A', border=1, align='C', fill=True)

                sev_color = severity_colors.get(v.severity, (255, 255, 255))
                pdf.set_fill_color(*sev_color)
                pdf.cell(30, 7, v.severity, border=1, align='C', fill=True)
                pdf.ln()
        else:
            pdf.set_font('Helvetica', 'I', 10)
            pdf.set_text_color(128, 128, 128)
            pdf.cell(0, 8, '  No violations recorded during this session.', ln=True)

        pdf.ln(10)

        # ── SECTION 6: Violation Summary (Counts by Type) ─────────────────────
        if violations:
            pdf.set_font('Helvetica', 'B', 12)
            pdf.set_text_color(15, 23, 42)
            pdf.cell(0, 8, 'VIOLATION SUMMARY BY TYPE', ln=True)
            pdf.ln(2)

            # Count violations by type
            type_counts = {}
            for v in violations:
                type_counts[v.violation_display] = type_counts.get(v.violation_display, 0) + 1

            pdf.set_font('Helvetica', '', 10)
            for vtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                bar_width = min(int(count / max(type_counts.values()) * 100), 100)
                pdf.cell(70, 7, f'  {vtype}:', fill=False)
                pdf.set_fill_color(99, 179, 237)
                pdf.cell(bar_width, 6, '', fill=True)
                pdf.cell(0, 7, f' {count}', ln=True)

        pdf.ln(10)

        # ── SECTION 7: Conclusion & Signature ─────────────────────────────────
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 8, 'EXAMINER NOTES', ln=True)
        pdf.ln(2)

        pdf.set_font('Helvetica', '', 10)
        pdf.set_fill_color(248, 250, 252)
        pdf.multi_cell(0, 7,
            f'This report was automatically generated by ExamGuard AI on '
            f'{datetime.utcnow().strftime("%B %d, %Y at %H:%M UTC")}.\n\n'
            f'The AI proctoring system analyzed video frames from the student\'s webcam '
            f'during the examination session. All violations have been timestamped and '
            f'recorded in the database with corresponding evidence screenshots.\n\n'
            f'This report should be reviewed by a qualified examiner before any '
            f'disciplinary action is taken.',
            border=1, fill=True
        )

        pdf.ln(10)
        pdf.cell(90, 8, 'Examiner Signature: ___________________', border=0)
        pdf.cell(90, 8, f'Date: {datetime.utcnow().strftime("%Y-%m-%d")}', border=0)

        # Save the PDF
        pdf.output(output_path)
        logger.info(f"PDF report generated: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        return ''
