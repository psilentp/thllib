split_dict = {'S-28':{'I':'?',
					  'II':'P{w[+mC]=BJD115F05-p65ADzpUw}attP40',
					  'III':'P{w[+mC]=GMR48E11-ZpGal4DBDUw}attP2',
					  'old_name':'DN106',
					  'new_name':'P10'},
			  'S-67':{'I':'?',
					  'II':'P{w[+mC]=BJD122B03-pBPp65ADZpUw}attP40',
					  'III':'P{w[+mC]=GMR22D06-pBPZpGdbdUw}attP2',
					  'old_name':'DN066',
					  'new_name':'G25'},
			  'S-61':{'I':'?',
					  'II':'P{w[+mC]=BJD105D02-pBPp65ADZpUw}attP40',
					  'III':'P{w[+mC]=GMR24C07-pBPZpGdbdUw}attP2',
					  'old_name':'DN010',
					  'new_name':'G24'},
			  'S-27':{'I':'?',
					  'II':'P{w[+mC]=BJD110A11-p65ADzpUw}attP40',
					  'III':'P{w[+mC]=BJD115F05-ZpGal4DBDUw}attP2',
					  'old_name':'DN106',
					  'new_name':'P10'},
			  'S-66':{'I':'?',
					  'II':'P{w[+mC]=BJD113B12-pBPp65ADZpUw}attP40',
					  'III':'P{w[+mC]=BJD110E05-pBPZpGdbdUw}attP2',
					  'old_name':'DN006',
					  'new_name':'B5'},
			  'S-56':{'I':'?',
					  'II':'P{w[+mC]=BJD137F08-pBPp65ADZpUw}attP40',
					  'III':'P{w[+mC]=BJD103C12-pBPZpGdbdUw}attP2',
					  'old_name':'DN165',
					  'new_name':'G17'},
			  'S-114':{'I':'?',
					  'II':'P{w[+mC]=GMR13D04-pBPp65ADZpUw}attP40',
					  'III':'P{w[+mC]=GMR65D05-pBPZpGdbdUw}attP2',
					  'old_name':'DN114',
					  'new_name':'G27'},
			  'S-204':{'I':'?',
					  'II':'P{w[+mC]=BJD104B12-pBPp65ADZpUw}attP40',
					  'III':'P{w[+mC]=GMR80H02-pBPZpGdbdUw}attP2',
					  'old_name':'DN094',
					  'new_name':'A8'},
			  'S-24':{'I':'?',
					  'II':'P{w[+mC]=GMR42B02-p65ADzpUw}attP40',
					  'III':'P{w[+mC]=BJD119F07-ZpGal4DBDUw}attP2',
					  'old_name':'DN045b',
					  'new_name':'G3'},
			  'S-249':{'I':'?',
					  'II':'P{w[+mC]=BJD111B02-pBPp65ADZpUw}attP40',
					  'III':'P{w[+mC]=BJD110C03-pBPZpGdbdUw}attP2',
					  'old_name':'DN045',
					  'new_name':'G2'},
			  'S-153':{'I':'?',
					  'II':'P{w[+mC]=GMR29F12-pBPp65ADZpUw}attP40',
					  'III':'P{w[+mC]=GMR37G07-pBPZpGdbdUw}attP2',
					  'old_name':'DN032a',
					  'new_name':'P3'},
			  'S-117':{'I':'?',
					  'II':'P{w[+mC]=GMR31H10-pBPp65ADZpUw}attP40',
					  'III':'P{w[+mC]=BJD111C04-pBPZpGdbdUw}attP2',
					  'old_name':'DN101a',
					  'new_name':'A7'}}


cs_fly =              {'I':'w[+]',
					   'II':'P{y[+t7.7] w[+mC]=13XLexAop2-IVS-GCaMP6f-p10}su(Hw)attP5, P{y[+t7.7] w[+mC]=GMR39E01-lexA}attP40',
					   'III': 'P{20XUAS-IVS-CsChrimson.mVenus}attP2'
}

chrimson_genotypes = dict()

for key,fly1 in split_dict.items():
	chrimson_genotypes[key] = {'I':fly1['I'] + '/' + cs_fly['I'],
	                           'II': fly1['II'] + '/' + cs_fly['II'],
	                           'III': fly1['III'] + '/' + cs_fly['III']}

def expand_snum(snum):
	fly = chrimson_genotypes[snum]
	s_string = ';'.join([fly[k] for k in ['I','II','III']])
	return s_string