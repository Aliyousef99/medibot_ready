from lab_parser import preclean, extract_bracket_ref, find_value_and_flag, find_unit, extract_name, canonicalize_name
line = 'CholesterolTotal(mmol/L) 4.7. mmol/L [<=5.2]'
cl = preclean(line)
print('CLEAN:', cl)
working, ref = extract_bracket_ref(cl)
print('WORKING:', working, 'REF:', ref)
vf = find_value_and_flag(working)
print('VF:', vf)
if vf:
    value, flag, span = vf
    name_part = extract_name(working[: span[0]])
    unit_part = find_unit(working[span[1] :], name_part)
    print('NAME_PART:', name_part)
    print('UNIT_PART:', unit_part)
    print('CANON:', canonicalize_name(name_part))
