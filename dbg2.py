from lab_parser import preclean, split_lines
raw = open('sample_labs.txt','r',encoding='utf-8').read()
cl = preclean(raw)
print('CLEANED:\n' + cl)
lines = split_lines(cl)
print('\nLINES:', lines)
print('CANDIDATE:', [ln for ln in lines if __import__('re').search(r'\d', ln)])
