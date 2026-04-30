from ucimlrepo import fetch_ucirepo

d = fetch_ucirepo(id=296)
X = d.data.features

print('A1Cresult:')
print(X['A1Cresult'].value_counts())

print()
print('change:')
print(X['change'].value_counts())

print()
print('max_glu_serum:')
print(X['max_glu_serum'].value_counts())
