#include <iostream>
#include <algorithm>
using namespace std;

const char Adrian[] = { 'A', 'B', 'C' };
const char Bruno[] = { 'B', 'A', 'B', 'C' };
const char Goran[] = { 'C', 'C', 'A', 'A', 'B', 'B' };

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N;
	string s;
	cin >> N >> s;

	int aCnt = 0, bCnt = 0, gCnt = 0;
	for (int i = 0; i < s.size(); i += 12) {
		for (int l = 0; l < 12; l++) {
			if (!(i + l < s.size())) break;

			if (s[i + l] == Adrian[l % 3]) aCnt++;
			if (s[i + l] == Bruno[l % 4]) bCnt++;
			if (s[i + l] == Goran[l % 6]) gCnt++;
		}
	}

	int ans = max({ aCnt, bCnt, gCnt });
	
	cout << ans << '\n';
	if (ans == aCnt) cout << "Adrian\n";
	if (ans == bCnt) cout << "Bruno\n";
	if (ans == gCnt) cout << "Goran\n";

	return 0;
}