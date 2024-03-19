#include <iostream>
#include <vector>
#include <queue>
using namespace std;

int N, K, picked[10];
string s, ans;

void findMax(int st, int digit);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> N >> K >> s;

	findMax(0, N - K);

	cout << ans;

	return 0;
}

void findMax(int st, int digit) {
	if (digit == 0) return;

	for (int i = st; i < s.size() - (digit - 1); i++) {
		for (int l = s[i] - '0' - 1; l >= 0; l--) {
			picked[l] = 0;
		}
		picked[s[i] - '0']++;
	}
	
	for (int i = 9; i >= 0; i--) {
		if (picked[i]) {
			ans.push_back(i + '0');
			picked[i]--;
			break;
		}
	}

	findMax(s.size() - (digit - 1), digit - 1);
}