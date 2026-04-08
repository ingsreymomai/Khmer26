export interface ExportMetadata {
  author?: string;
  date?: string;
  title?: string;
}

export const exportToWord = async (
  htmlContent: string, 
  filename: string, 
  headerHtml: string = '', 
  marginValue: string = '0.4in 0.6in 0.4in 0.6in',
  fontFamily: string = 'Times New Roman',
  lineHeight: string = '1.15',
  metadata?: ExportMetadata,
  isFrameEnabled: boolean = false
) => {
  const tempDiv = document.createElement('div');
  tempDiv.innerHTML = htmlContent;

  const linePercentage = `${parseFloat(lineHeight) * 100}%`;
  const exactLineHeight = `${(parseFloat(lineHeight) * 12).toFixed(1)}pt`;

  // 1. FIX: Convert all images to Base64 (This prevents "Empty Boxes")
  const images = tempDiv.querySelectorAll('img');
  for (const img of Array.from(images)) {
    try {
      const response = await fetch(img.src);
      const blob = await response.blob();
      const base64 = await new Promise((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.readAsDataURL(blob);
      });
      img.src = base64 as string;
      
      // Force fixed size so they don't overlap
      // We use inches for Word compatibility
      const originalWidth = img.width || 550;
      if (originalWidth > 200) {
        // Large images (like Quest Lab images)
        img.setAttribute('width', '550'); 
        img.style.width = '6.5in'; // Adjusted for 1in margins (8.5 - 2 = 6.5)
        img.style.height = 'auto';
      } else if (originalWidth < 50) {
        // Small icons
        img.setAttribute('width', '45');
        img.style.width = '0.45in';
        img.style.height = 'auto';
      } else {
        // Medium images - preserve relative size
        const inWidth = (originalWidth / 96).toFixed(2);
        img.setAttribute('width', originalWidth.toString());
        img.style.width = `${inWidth}in`;
        img.style.height = 'auto';
      }
      img.style.display = 'block';
      img.style.margin = '5px auto';
    } catch (e) {
      console.warn("Could not convert image to base64", e);
    }
  }

  // 2. FIX: Handle Round MCQ Badges for Word
  const worksheetPage = tempDiv.querySelector('.worksheet-page');
  const designClass = worksheetPage ? Array.from(worksheetPage.classList).find(c => c.startsWith('design-')) : '';

  const badges = tempDiv.querySelectorAll('b, strong');
  badges.forEach(badge => {
    const isOptionLetter = badge.textContent?.trim().length === 1 && /^[A-D]$/.test(badge.textContent.trim());
    if (isOptionLetter) {
      (badge as HTMLElement).style.display = 'inline-block';
      (badge as HTMLElement).style.width = '22pt';
      (badge as HTMLElement).style.height = '22pt';
      (badge as HTMLElement).style.lineHeight = '22pt';
      (badge as HTMLElement).style.textAlign = 'center';
      (badge as HTMLElement).style.marginRight = '6pt';
      (badge as HTMLElement).style.fontWeight = 'bold';
      (badge as HTMLElement).style.fontSize = '10pt';
      (badge as HTMLElement).style.verticalAlign = 'middle';

      // Design-Specific Word Fallbacks
      if (designClass === 'design-modern-blue') {
        (badge as HTMLElement).style.border = '1.5pt solid #2563eb';
        (badge as HTMLElement).style.backgroundColor = '#eff6ff';
        (badge as HTMLElement).style.color = '#1e40af';
      } else if (designClass === 'design-classic') {
        (badge as HTMLElement).style.border = '1pt solid black';
        (badge as HTMLElement).style.backgroundColor = 'transparent';
        (badge as HTMLElement).style.borderRadius = '0';
      } else if (designClass === 'design-playful') {
        (badge as HTMLElement).style.border = '2pt solid #f97316';
        (badge as HTMLElement).style.backgroundColor = '#ffedd5';
        (badge as HTMLElement).style.color = '#9a3412';
        (badge as HTMLElement).style.borderRadius = '11pt';
      } else if (designClass === 'design-technical') {
        (badge as HTMLElement).style.backgroundColor = '#0f172a';
        (badge as HTMLElement).style.color = '#ffffff';
        (badge as HTMLElement).style.border = 'none';
      } else if (designClass === 'design-elegant') {
        (badge as HTMLElement).style.border = '1pt solid #92400e';
        (badge as HTMLElement).style.backgroundColor = '#fef3c7';
      } else if (designClass === 'design-contrast') {
        (badge as HTMLElement).style.backgroundColor = 'black';
        (badge as HTMLElement).style.color = 'white';
        (badge as HTMLElement).style.border = 'none';
      } else if (designClass === 'design-modern-round' || designClass === 'design-projector') {
        // Force a "Circle" look as much as possible in Word
        (badge as HTMLElement).style.border = '1.5pt solid #6366f1';
        (badge as HTMLElement).style.backgroundColor = '#e0e7ff';
        (badge as HTMLElement).style.color = '#4338ca';
        (badge as HTMLElement).style.borderRadius = '11pt';
      } else {
        // Default
        (badge as HTMLElement).style.border = '2pt solid #475569'; // Thicker, darker gray for better visibility in Word
        (badge as HTMLElement).style.backgroundColor = '#f8fafc';
        (badge as HTMLElement).style.borderRadius = '11pt';
      }
    }
  });

  // 2.1 FIX: Handle MCQ Blank Start and Underlined Letters for Word
  const specialElements = tempDiv.querySelectorAll('.mcq-blank-start, u, .blank-line, .checkbox-box');
  specialElements.forEach(el => {
    if (el.classList.contains('mcq-blank-start') || el.classList.contains('blank-line')) {
      (el as HTMLElement).style.display = 'inline-block';
      (el as HTMLElement).style.width = '50pt';
      (el as HTMLElement).style.borderBottom = '1pt solid black';
      (el as HTMLElement).style.marginRight = '10pt';
      (el as HTMLElement).style.textAlign = 'center';
      (el as HTMLElement).style.textDecoration = 'none';
      (el as HTMLElement).innerHTML = '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;';
    }
    if (el.classList.contains('checkbox-box')) {
      (el as HTMLElement).style.display = 'inline-block';
      (el as HTMLElement).style.width = '12pt';
      (el as HTMLElement).style.height = '12pt';
      (el as HTMLElement).style.border = '1pt solid black';
      (el as HTMLElement).style.marginRight = '5pt';
      (el as HTMLElement).style.verticalAlign = 'middle';
    }
    // If it's a <u> used for the "Letter on Line" style
    if (el.tagName === 'U' && el.textContent?.includes('\u00A0')) {
      (el as HTMLElement).style.borderBottom = '1pt solid black';
      (el as HTMLElement).style.textDecoration = 'none';
      (el as HTMLElement).style.padding = '0 5pt';
    }
  });

  // 3. FIX: Wrap everything in 100% Tables (This stops things from overlapping)
  // This is the "Magic Fix" for Word
  const sections = tempDiv.children;
  let finalHtml = "";
  for (let i = 0; i < sections.length; i++) {
    const el = sections[i];
    // If it's a new Set title, force a page break
    const isNewSet = el.textContent?.toUpperCase().includes('(SET');
    const pageBreak = isNewSet && i > 0 ? 'style="page-break-before:always"' : '';
    
    finalHtml += `
      <table border="0" cellspacing="0" cellpadding="0" width="100%" ${pageBreak} style="margin: 0; padding: 0; border-collapse: collapse; mso-table-lspace:0pt; mso-table-rspace:0pt;">
        <tr>
          <td align="left" style="padding: 0; margin: 0; font-family: '${fontFamily}', serif; font-size: 12pt; line-height: ${exactLineHeight}; mso-line-height-rule: exactly;">
            ${el.outerHTML}
          </td>
        </tr>
      </table>`;
  }

  // 3. FIX: Table Formatting (Ensures grids and nested MCQ tables are correct)
  const tables = tempDiv.querySelectorAll('table');
  tables.forEach(table => {
    const isNested = table.parentElement?.closest('table') !== null;
    
    if (isNested) {
      // Nested tables (like MCQ options) should have NO borders and 100% width
      table.setAttribute('border', '0');
      table.style.border = 'none';
      table.style.width = '100%';
      table.style.borderCollapse = 'collapse';
      const cells = table.querySelectorAll('td');
      cells.forEach((c) => {
        (c as HTMLElement).style.border = 'none';
        (c as HTMLElement).style.padding = '2pt';
        
        // Check if it's the first cell in its row
        const isFirstInRow = c.previousElementSibling === null;
        if (isFirstInRow) {
          (c as HTMLElement).style.paddingLeft = '30pt'; // Indent "A." by approx 7 spaces
        }
        (c as HTMLElement).style.verticalAlign = 'top';
        (c as HTMLElement).style.width = '25%'; // Default for 4-column MCQ
      });
    } else {
      // Top-level tables should have borders
      table.setAttribute('border', '1');
      table.style.borderCollapse = 'collapse';
      table.style.margin = '0 auto';
      table.style.width = '100%';
      
      // Check if it's a Word Search (many small cells with single letters)
      const cells = table.querySelectorAll('td');
      const isWordSearch = cells.length > 20 && Array.from(cells).every(c => (c.textContent?.trim().length || 0) <= 1);
      
      if (isWordSearch) {
        cells.forEach(c => {
          (c as HTMLElement).style.width = '25pt';
          (c as HTMLElement).style.height = '25pt';
          (c as HTMLElement).style.textAlign = 'center';
          (c as HTMLElement).style.lineHeight = exactLineHeight;
          (c as HTMLElement).style.setProperty('mso-line-height-rule', 'exactly');
        });
      }
    }
  });

  let metadataHtml = "";
  if (metadata) {
    metadataHtml = `
      <div style="margin-bottom: 20pt; border-bottom: 1pt solid #ccc; padding-bottom: 10pt; font-size: 9pt; color: #666; font-family: '${fontFamily}', serif;">
        ${metadata.title ? `<div style="font-size: 14pt; font-weight: bold; color: #000; margin-bottom: 5pt;">${metadata.title}</div>` : ''}
        ${metadata.author ? `<div><strong>Author:</strong> ${metadata.author}</div>` : ''}
        ${metadata.date ? `<div><strong>Date:</strong> ${metadata.date}</div>` : `<div><strong>Exported on:</strong> ${new Date().toLocaleDateString()}</div>`}
      </div>
    `;
  }

  const mTop = '0.4in';
  const mRight = '0.6in';
  const mBottom = '0.4in';
  const mLeft = '0.6in';

  const frameStyle = isFrameEnabled ? 'border: 1.5pt solid black; padding: 10pt;' : '';

  const content = `
    <html xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:w="urn:schemas-microsoft-com:office:word" xmlns="http://www.w3.org/TR/REC-html40">
    <head>
      <meta charset='utf-8'>
      <style>
        @page Section1 { 
          size: 8.5in 11.0in; 
          margin: ${mTop} ${mRight} ${mBottom} ${mLeft}; 
          mso-header-margin: 0.5in; 
          mso-footer-margin: 0.5in; 
          mso-paper-source: 0;
          mso-page-margin-top: ${mTop};
          mso-page-margin-bottom: ${mBottom};
          mso-page-margin-left: ${mLeft};
          mso-page-margin-right: ${mRight};
        }
        div.Section1 { page: Section1; ${frameStyle} }
        body { 
          font-family: "${fontFamily}", serif; 
          font-size: 12pt; 
          line-height: ${exactLineHeight}; 
          mso-line-height-rule: exactly; 
          margin: 0;
          padding: 0;
        }
        p, div, li, span, ol, ul { 
          margin: 0pt; 
          padding: 0pt; 
          line-height: ${exactLineHeight}; 
          mso-line-height-rule: exactly;
          mso-ascii-font-family: "${fontFamily}";
          mso-hansi-font-family: "${fontFamily}";
          mso-bidi-font-family: "${fontFamily}";
          mso-margin-top-alt: 0pt;
          mso-margin-bottom-alt: 0pt;
        }
        img { border: none; display: block; }
        table { 
          mso-table-lspace:0pt; 
          mso-table-rspace:0pt; 
          border-collapse: collapse; 
          margin: 0; 
          width: 100%;
          mso-line-height-rule: exactly;
        }
        td { 
          font-family: "${fontFamily}", serif; 
          font-size: 12pt; 
          line-height: ${exactLineHeight}; 
          mso-line-height-rule: exactly; 
          padding: 0;
          vertical-align: top;
          mso-margin-top-alt: 0pt;
          mso-margin-bottom-alt: 0pt;
        }
        .header-row { background-color: #334155; color: white; text-align: center; font-weight: bold; padding: 10px; }
      </style>
    </head>
    <body>
      <div class="Section1">
        ${headerHtml}
        ${metadataHtml}
        ${finalHtml}
      </div>
    </body>
    </html>`;

  const blob = new Blob(['\ufeff', content], { type: 'application/msword' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${filename}.doc`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};