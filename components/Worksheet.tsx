import React, { useMemo, useRef, useEffect } from 'react';
import { AcademicLevel, Theme, PaperType, BrandSettings } from '../types';

interface WorksheetProps {
  content: string;
  onContentChange: (val: string) => void;
  isGenerating: boolean;
  level: AcademicLevel;
  module: string;
  topic: string;
  paperType: PaperType;
  theme: Theme;
  brandSettings: BrandSettings;
  paperDesign: number;
  isRoundMcq: boolean;
  isColorfulBackgroundEnabled: boolean;
  isInstructionBackgroundEnabled: boolean;
  zoom?: number;
}

const Worksheet: React.FC<WorksheetProps> = ({
  content, onContentChange, isGenerating, brandSettings, paperDesign, isRoundMcq, isColorfulBackgroundEnabled, isInstructionBackgroundEnabled, zoom = 100
}) => {
  const editorRef = useRef<HTMLDivElement>(null);

  const designClasses = [
    '', // 0: Standard
    'design-modern-blue', // 1
    'design-classic', // 2
    'design-minimalist', // 3
    'design-playful', // 4
    'design-professional', // 5
    'design-elegant', // 6
    'design-technical', // 7
    'design-eco', // 8
    'design-contrast', // 9
    'design-two-fold', // 10
    'design-projector', // 11
    'design-modern-round', // 12
  ];

  const placeholderHtml = useMemo(() => {
    return `
      <div class="flex flex-col items-center h-full min-h-[600px] text-center px-4 py-20 relative overflow-hidden bg-white text-black">
        <div class="border-[12px] border-black p-12 max-w-2xl w-full">
            <div class="space-y-6 text-center">
                <h1 class="text-2xl md:text-3xl font-black uppercase tracking-widest">${brandSettings.schoolName}</h1>
                <hr class="border-black border-2 w-full" />
                <p class="text-sm md:text-lg font-bold italic tracking-[0.3em] uppercase">${brandSettings.schoolAddress}</p>
            </div>
        </div>
        <div class="mt-20 w-full max-w-2xl text-left font-bold text-[10px] space-y-4">
            <div class="flex justify-between border-b border-slate-200 pb-2 text-slate-400">
                <span>NAME : _________________________________</span>
                <span>DATE : ____ / ____ / ____</span>
            </div>
        </div>
      </div>
    `;
  }, [brandSettings.schoolName, brandSettings.schoolAddress]);

  useEffect(() => {
    if (editorRef.current && content !== editorRef.current.innerHTML) {
      if (!isGenerating) {
        editorRef.current.innerHTML = content || placeholderHtml;
      }
    }
  }, [content, isGenerating, placeholderHtml]);

  return (
    <div className="flex-1 overflow-auto p-2 md:p-12 no-scrollbar bg-slate-200/50">
      <div className="w-full max-w-[210mm] mx-auto pb-64 shadow-2xl worksheet-paper transition-transform duration-300 ease-in-out" style={{ transform: `scale(${zoom / 100})`, transformOrigin: 'top center' }}>
        <style>{`
          .prose { 
            font-family: '${brandSettings.activeFont || 'Times New Roman'}', serif !important; 
            font-size: ${brandSettings.fontSize || 12}pt !important; 
            line-height: 1.15 !important;
            padding: 0.4in 0.5in !important;
          }
          @media (min-width: 768px) {
            .prose { padding: 0.8in 1in !important; }
          }
          .prose p, .prose div { margin-bottom: 8pt !important; }
          .prose div[style*="background"] { padding: 15pt !important; border-radius: 8pt; margin-bottom: 20pt !important; }
          .prose li, .prose ol, .prose ul { margin: 0 !important; padding: 0 !important; }
          .prose table { border-collapse: collapse !important; width: 100% !important; border: 1.5pt solid black !important; table-layout: fixed; margin-bottom: 0 !important; }
          .prose table table { border: none !important; margin-top: 5pt !important; width: auto !important; }
          .prose table table td { border: none !important; padding: 2pt 10pt !important; }
          .prose table table tr td:first-child { padding-left: 30pt !important; }
          .prose .options-table { width: 100% !important; table-layout: fixed !important; }
          .prose .options-table td { padding-left: 30pt !important; }
          .prose .options-table tr td:first-child { padding-left: 30pt !important; }
          .prose .options-table tr td:not(:first-child) { padding-left: 10pt !important; }
          .prose .blank-line { border-bottom: 1pt solid black; display: inline-block; min-width: 50pt; margin: 0 5pt; }
          .prose .checkbox-box { border: 1pt solid black; width: 12pt; height: 12pt; display: inline-block; margin-right: 5pt; vertical-align: middle; }
          .prose th, .prose td { border: 1pt solid black !important; padding: 6pt !important; vertical-align: top !important; }
          .prose .header-row, .prose tr:first-child td[colspan] { text-align: center !important; font-weight: bold !important; padding: 6px !important; }
          .prose .header-row:not([style*="background"]), .prose tr:first-child td[colspan]:not([style*="background"]) { background-color: #334155; color: white; }

          /* Zebra Striping for professional tables */
          .prose tr:nth-child(even) td { background-color: #f8fafc; }
          .prose tr:nth-child(odd) td { background-color: #ffffff; }
          .prose .header-row td, .prose tr:first-child td { background-color: inherit; }
          .prose .options-table tr td { background-color: transparent !important; }
          
          .prose .professional-table {
            border-collapse: collapse !important;
            width: 100% !important;
            margin-bottom: 20pt !important;
          }
          .prose .professional-table thead th {
            background-color: #334155 !important;
            color: white !important;
            text-align: center !important;
            font-weight: bold !important;
            padding: 10pt !important;
          }
          .prose .professional-table tbody td {
            padding: 8pt !important;
            border: 1pt solid #e2e8f0 !important;
          }

          /* Teacher Answer Key Styling */
          .prose .answer-key-section {
            margin-top: 40pt !important;
            padding: 20pt !important;
            border-top: 2pt dashed #cbd5e1 !important;
            background-color: #f8fafc !important;
            border-radius: 8pt !important;
          }
          .prose .answer-key-section h2 {
            color: #1e293b !important;
            font-size: 14pt !important;
            text-transform: uppercase !important;
            letter-spacing: 0.1em !important;
            margin-bottom: 10pt !important;
          }
          .prose .answer-key-section b, .prose .answer-key-section strong {
            border: none !important;
            background: transparent !important;
            min-width: auto !important;
            height: auto !important;
            display: inline !important;
            margin-right: 0.2em !important;
            font-size: inherit !important;
          }

          /* Paper Designs */
          .design-modern-blue { border: 4pt solid #2563eb !important; border-radius: 0 !important; box-shadow: 15pt 15pt 0 #dbeafe !important; }
          .design-modern-blue .header-row { background-color: #2563eb !important; color: white !important; text-transform: uppercase; }
          
          .design-classic { border: 1.5pt solid black !important; padding: 1.2in !important; outline: 4pt double black !important; outline-offset: -15pt !important; }
          .design-classic .header-row { background-color: black !important; color: white !important; font-family: 'Georgia', serif !important; }
          
          .design-minimalist { border: none !important; box-shadow: none !important; padding: 1.5in !important; }
          .design-minimalist .header-row { background: transparent !important; color: black !important; border-bottom: 3pt solid black !important; text-transform: uppercase; letter-spacing: 0.4em; font-weight: 900 !important; }
          
          .design-playful { border: 4pt dashed #f97316 !important; border-radius: 40pt !important; background-color: #fff7ed !important; }
          .design-playful .header-row { background-color: #f97316 !important; border-radius: 20pt 20pt 0 0 !important; font-size: 1.2em !important; }
          .design-playful td { border: 2pt dashed #fdba74 !important; border-radius: 15pt !important; }
          
          .design-professional { border-left: 25pt solid #1e293b !important; border-right: 1pt solid #e2e8f0 !important; border-top: 1pt solid #e2e8f0 !important; border-bottom: 1pt solid #e2e8f0 !important; background-color: #f8fafc !important; }
          .design-professional .header-row { background-color: #334155 !important; text-align: left !important; padding-left: 20pt !important; }
          
          .design-elegant { border: 1pt solid #92400e !important; background-color: #fffbeb !important; position: relative; }
          .design-elegant::before { content: ''; position: absolute; inset: 10pt; border: 1pt solid #d97706; pointer-events: none; }
          .design-elegant .header-row { background-color: #92400e !important; font-family: 'Garamond', serif !important; font-style: italic !important; }
          
          .design-technical { border: 2pt solid #0f172a !important; background-image: linear-gradient(#e2e8f0 1px, transparent 1px), linear-gradient(90deg, #e2e8f0 1px, transparent 1px) !important; background-size: 30px 30px !important; }
          .design-technical .header-row { background-color: #0f172a !important; font-family: 'Courier New', monospace !important; }
          
          .design-eco { border: 6pt solid #166534 !important; border-radius: 60pt 10pt 60pt 10pt !important; background-color: #f0fdf4 !important; }
          .design-eco .header-row { background-color: #166534 !important; border-radius: 40pt 0 0 0 !important; }
          
          .design-contrast { border: 12pt solid black !important; padding: 0.5in !important; }
          .design-contrast .header-row { background-color: black !important; color: white !important; font-size: 1.5em !important; letter-spacing: -0.05em !important; }

          .design-two-fold { 
            border: 1pt solid #cbd5e1 !important; 
            background: linear-gradient(90deg, #ffffff 49.5%, #e2e8f0 50%, #ffffff 50.5%) !important;
            box-shadow: 0 20px 50px rgba(0,0,0,0.1) !important;
            padding: 1in !important;
          }
          .design-two-fold .header-row { background-color: #64748b !important; }

          .design-projector {
            background-color: #000000 !important;
            color: #ffffff !important;
            border: 5pt solid #ffffff !important;
            padding: 1.5in !important;
          }
          .design-projector .header-row { background-color: #ffffff !important; color: #000000 !important; font-size: 2em !important; font-weight: 900 !important; }
          .design-projector td { border: 1pt solid #ffffff !important; color: #ffffff !important; }
          
          .design-modern-round {
            border: 2pt solid #6366f1 !important;
            border-radius: 50pt !important;
            padding: 1.2in !important;
            background-color: #f5f3ff !important;
          }
          .design-modern-round .header-row { background-color: #6366f1 !important; border-radius: 40pt 40pt 0 0 !important; }

          /* MCQ Design Variety Support */
          .prose .mcq-blank-start {
            display: inline-block;
            margin-right: 10pt;
            border-bottom: 1pt solid black;
            min-width: 40pt;
            text-align: center;
          }
          .prose u {
            text-decoration: none !important;
            border-bottom: 1pt solid black;
            display: inline-block;
            min-width: 25pt;
            text-align: center;
            margin-right: 5pt;
          }

          /* Matching Variety Support */
          .prose .matching-blank {
            display: inline-block;
            width: 80pt;
            border-bottom: 1pt solid #94a3b8;
            margin-right: 10pt;
            vertical-align: bottom;
          }
          .prose .word-bank-box {
            border: 1.5pt solid #334155;
            padding: 15pt;
            margin: 15pt 0;
            background-color: #f8fafc;
            border-radius: 4pt;
            display: flex;
            flex-wrap: wrap;
            gap: 15pt;
            justify-content: center;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 0.05em;
          }
          .prose .word-bank-box-alt {
            border: 2pt double #1e293b;
            padding: 12pt;
            margin: 15pt 0;
            background-color: #ffffff;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(80pt, 1fr));
            gap: 10pt;
            text-align: center;
            font-family: 'Courier New', monospace;
          }
          .prose .word-bank-item {
            padding: 2pt 8pt;
          }
          .prose .word-bank-item-box {
            border: 1pt solid #cbd5e1;
            padding: 4pt;
            background: #f1f5f9;
          }

          /* Ruler Table Style */
          .prose .ruler-table {
            border-collapse: collapse !important;
            width: 100% !important;
            border: none !important;
            margin-bottom: 20pt !important;
          }
          .prose .ruler-table td {
            border: none !important;
            padding: 15pt !important;
            vertical-align: top !important;
          }
          .prose .ruler-table td:first-child {
            border-right: 1.5pt solid #334155 !important;
          }
          .prose .ruler-table .header-row td {
            border-right: none !important;
            border-bottom: 1.5pt solid #334155 !important;
          }

          /* Relaxing Part Backgrounds */
          .prose .bg-relax-blue { background-color: #f0f9ff !important; border-radius: 8pt; padding: 15pt !important; margin-bottom: 15pt !important; }
          .prose .bg-relax-green { background-color: #f0fdf4 !important; border-radius: 8pt; padding: 15pt !important; margin-bottom: 15pt !important; }
          .prose .bg-relax-yellow { background-color: #fffbeb !important; border-radius: 8pt; padding: 15pt !important; margin-bottom: 15pt !important; }
          .prose .bg-relax-purple { background-color: #f5f3ff !important; border-radius: 8pt; padding: 15pt !important; margin-bottom: 15pt !important; }
          .prose .bg-relax-rose { background-color: #fff1f2 !important; border-radius: 8pt; padding: 15pt !important; margin-bottom: 15pt !important; }
          .prose .bg-relax-ocean { 
            background-image: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 100%) !important; 
            border-left: 5pt solid #0ea5e9 !important;
            border-radius: 0 8pt 8pt 0;
            padding: 15pt !important; 
            margin-bottom: 15pt !important; 
          }
          .prose .bg-relax-mist { 
            background-image: linear-gradient(135deg, #f1f5f9 0%, #f8fafc 100%) !important; 
            border: 1pt solid #e2e8f0 !important;
            border-radius: 12pt;
            padding: 15pt !important; 
            margin-bottom: 15pt !important; 
          }
          .prose .bg-relax-forest { 
            background-image: linear-gradient(135deg, #dcfce7 0%, #f0fdf4 100%) !important; 
            border-right: 5pt solid #166534 !important;
            border-radius: 8pt 0 0 8pt;
            padding: 15pt !important; 
            margin-bottom: 15pt !important; 
          }

          /* Round MCQ Support - Base Styles */
          .prose b, .prose strong {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 1.2em;
            height: 1.2em;
            margin-right: 0.4em;
            transition: all 0.2s;
            /* Word Export Compatibility: Use standard borders */
            border: none;
            background: transparent;
          }

          /* Design-Specific Badge Overrides (Applied when isRoundMcq is true) */
          ${isRoundMcq ? `
            .design-modern-blue b, .design-modern-blue strong { background: #eff6ff; border: 0.5pt solid #2563eb; border-radius: 50% !important; color: #1e40af; }
            .design-classic b, .design-classic strong { background: transparent; border: 0.5pt solid black; border-radius: 50% !important; font-family: 'Georgia', serif; }
            .design-minimalist b, .design-minimalist strong { background: #f8fafc; border: none; border-bottom: 1pt solid black; border-radius: 0 !important; }
            .design-playful b, .design-playful strong { background: #ffedd5; border: 1pt dashed #f97316; border-radius: 50% !important; color: #9a3412; }
            .design-professional b, .design-professional strong { background: #f1f5f9; border-left: 3pt solid #334155; border-radius: 0 !important; padding-left: 4px; width: auto; min-width: 1.8em; }
            .design-elegant b, .design-elegant strong { background: #fef3c7; border: 0.5pt solid #92400e; border-radius: 50% 0 50% 0 !important; color: #78350f; }
            .design-technical b, .design-technical strong { background: #0f172a; color: white; border-radius: 0 !important; font-family: 'Courier New', monospace; }
            .design-eco b, .design-eco strong { background: #dcfce7; border: 1pt solid #166534; border-radius: 12px 4px !important; color: #14532d; }
            .design-contrast b, .design-contrast strong { background: black; color: white; border-radius: 0 !important; transform: rotate(-3deg); }
            .design-two-fold b, .design-two-fold strong { background: white; border: 0.5pt solid #cbd5e1; border-radius: 50% !important; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
            .design-projector b, .design-projector strong { background: white; color: black; border-radius: 50% !important; font-weight: 900 !important; }
            .design-modern-round b, .design-modern-round strong { background: #e0e7ff; border: 1pt solid #6366f1; border-radius: 50% !important; color: #4338ca; }
            
            /* Default if no design class matches but isRoundMcq is true */
            .worksheet-page:not([class*="design-"]) b, .worksheet-page:not([class*="design-"]) strong {
              background: transparent;
              border-radius: 50% !important;
              border: 0.5pt solid #cbd5e1;
              font-size: 0.9em;
              color: inherit;
              font-weight: 700 !important;
              box-shadow: none;
            }
          ` : `
            /* Plain style when Round is OFF */
            .prose b, .prose strong { border: none; background: transparent; min-width: auto; height: auto; display: inline; margin-right: 0.2em; }
          `}

          /* Design-Specific Table Overrides */
          .design-minimalist table { border: none !important; border-top: 2pt solid black !important; border-bottom: 2pt solid black !important; }
          .design-minimalist td, .design-minimalist th { border: none !important; border-bottom: 1pt solid #e2e8f0 !important; }
          
          .design-playful table { border: 3pt dashed #f97316 !important; border-radius: 20pt !important; overflow: hidden; }
          .design-playful td { border: 1.5pt dashed #fdba74 !important; }
          
          .design-technical table { border: 1.5pt solid #0f172a !important; font-family: 'Courier New', monospace !important; }
          .design-technical td { border: 1pt solid #334155 !important; }

          .design-elegant table { border: 1pt solid #92400e !important; }
          .design-elegant td { border: 0.5pt solid #d97706 !important; }

          .design-contrast table { border: 4pt solid black !important; }
          .design-contrast td { border: 2pt solid black !important; }

          .design-modern-blue table { border: 1.5pt solid #2563eb !important; }
          .design-modern-blue td { border: 0.5pt solid #bfdbfe !important; }
          .design-modern-blue tr:nth-child(even) td { background-color: #eff6ff; }

          .design-classic table { border: 2pt solid black !important; outline: 0.5pt solid black !important; outline-offset: 2pt !important; }
          .design-classic td { border: 0.5pt solid black !important; font-family: 'Georgia', serif !important; }

          .design-professional table { border: 1pt solid #334155 !important; border-left: 6pt solid #334155 !important; }
          .design-professional td { border: 0.5pt solid #e2e8f0 !important; }

          .design-eco table { border: 2pt solid #166534 !important; border-radius: 10pt !important; overflow: hidden; }
          .design-eco td { border: 1pt solid #bbf7d0 !important; background-color: #f0fdf4; }

          .design-modern-round table { border: 2pt solid #6366f1 !important; border-radius: 15pt !important; overflow: hidden; }
          .design-modern-round td { border: 1pt solid #e0e7ff !important; }
          .design-modern-round tr:nth-child(even) td { background-color: #f5f3ff; }

          .design-modern-blue .professional-table thead th { background-color: #2563eb !important; }
          .design-classic .professional-table thead th { background-color: black !important; font-family: 'Georgia', serif !important; }
          .design-playful .professional-table thead th { background-color: #f97316 !important; border-radius: 10pt 10pt 0 0 !important; }
          .design-technical .professional-table thead th { background-color: #0f172a !important; font-family: 'Courier New', monospace !important; }
          .design-eco .professional-table thead th { background-color: #166534 !important; }
          .design-modern-round .professional-table thead th { background-color: #6366f1 !important; border-radius: 15pt 15pt 0 0 !important; }

          .worksheet-footer {
            margin-top: 40pt;
            border-top: 1pt solid #e2e8f0;
            padding-top: 10pt;
            text-align: center;
            font-size: 8pt;
            color: #94a3b8;
            font-style: italic;
          }

          @media print {
            .no-print { display: none !important; }
            .bg-white { background-color: white !important; }
          }
        `}</style>
        
        <div className={`worksheet-page prose min-h-[297mm] p-[0.8in_1in] relative rounded-sm ${designClasses[paperDesign] || ''} ${isColorfulBackgroundEnabled ? 'bg-blue-50/50' : 'bg-white'}`}
             ref={editorRef}
             contentEditable={!isGenerating}
             onInput={(e) => onContentChange(e.currentTarget.innerHTML)}
             dangerouslySetInnerHTML={{ __html: content || placeholderHtml }}
        />
      </div>
    </div>
  );
};

export default Worksheet;
