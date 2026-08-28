import jsPDF from 'jspdf';

// x/y coordinates below are manual point positions on a fixed A4-ish page,
// not a layout system — this is a fixed-format single-page report, so a
// static coordinate grid is simpler than a flowing layout.
export function compilePdfReport(report) {
  const pdf = new jsPDF();

  const {
    filename,
    verdict,
    confidence,
    probability,
    type,
    file_size_bytes,
    processing_time_sec,
    view_agreement,
    is_low_confidence,
  } = report;

  pdf.setFont('courier', 'bold');
  pdf.setFontSize(22);
  pdf.text('ViT-CORE-Audio Anti-Spoofing Report', 20, 20);

  pdf.setFontSize(12);
  pdf.setFont('courier', 'normal');
  pdf.text(`Generated: ${new Date().toLocaleString()}`, 20, 30);
  pdf.line(20, 35, 190, 35);

  pdf.setFont('courier', 'bold');
  pdf.text('Media File Details', 20, 45);
  pdf.setFont('courier', 'normal');
  pdf.text(`Filename: ${filename}`, 20, 55);
  pdf.text(`Format: ${String(type).toUpperCase()}`, 20, 65);
  pdf.text(`File Size: ${(file_size_bytes / 1024).toFixed(1)} KB`, 20, 75);

  pdf.setFont('courier', 'bold');
  pdf.text('Analysis Verdict', 20, 95);
  pdf.setFont('courier', 'normal');
  pdf.setTextColor(verdict === 'SPOOF' ? 255 : 0, 0, verdict === 'BONAFIDE' ? 255 : 0);
  pdf.text(`Verdict: ${verdict}`, 20, 105);
  pdf.setTextColor(0, 0, 0);
  pdf.text(`Confidence: ${confidence}%`, 20, 115);
  pdf.text(`Raw Spoof Probability: ${probability}`, 20, 125);

  pdf.setFont('courier', 'bold');
  pdf.text('Model Telemetry', 20, 145);
  pdf.setFont('courier', 'normal');
  pdf.text(`Mel/CQT View Agreement: ${view_agreement}`, 20, 155);
  pdf.text(`Analysis Window: 4.0s fixed @ 16kHz`, 20, 165);
  pdf.text(`Processing Time: ${processing_time_sec} sec`, 20, 175);
  pdf.text(`Ambiguity Flag: ${is_low_confidence ? 'FLAGGED - MANUAL REVIEW' : 'Clear'}`, 20, 185);

  pdf.setFontSize(10);
  pdf.setTextColor(100, 100, 100);
  pdf.text('Disclaimer: Results are probabilistic and should be corroborated with other evidence.', 20, 280);

  const safeName = (filename || 'report')
    .replace(/[^\w.-]/g, '_')
    .replace(/\.[^.]+$/, '');

  pdf.save(`${safeName}_audio_forensics_report.pdf`);
}
