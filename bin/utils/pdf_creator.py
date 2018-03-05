from fpdf import FPDF


class PDF(FPDF):
    def header(self):
        # Logo
        self.image('utils/aequitas_report_header.png', x=5, w=0, h=20)
        self.ln(10)
        # Arial bold 15
        self.set_font('Arial', 'B', 20)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(30, 10, 'Bias and Fairness Audit Report', 0, 0, 'C')
        # Line break
        self.ln(10)

    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, 'aequitas \xa9 2017. The University of Chicago. All Rights '
                         'Reserved.     '
                         '\t \t'
                         'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

# fpdf.cell(w, h = 0, txt = '', border = 0, ln = 0,
#          align = '', fill = False, link = '')
