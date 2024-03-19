#include <bits/stdc++.h>
using namespace std;

const int MAX = 1e5;

int a1[MAX + 5], a2[MAX + 5];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		int N; cin >> N;

		for (int i = 0; i < N; i++)
			cin >> a1[i];
		for (int i = 0; i < N; i++)
			cin >> a2[i];

		bool plus = false, minus = false, YES = true;

		for (int i = 0; i < N; i++) {
			if (a1[i] > a2[i] && !minus) YES = false;
			else if (a1[i] < a2[i] && !plus) YES = false;

			if (a1[i] == 1) plus = true;
			if (a1[i] == -1) minus = true;
		}

		if (YES) cout << "YES" << endl;
		else cout << "NO" << endl;
	}

	return 0;
}