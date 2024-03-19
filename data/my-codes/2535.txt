#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
#define vi vector<int>
#define vl vector<ll>
#define vb vector<bool>
#define vd vector<double>
#define vc vector<char>
#define vs vector<string>
#define vpi vector<pii>
#define vpl vector<pll>
#define vvi vector<vector<int>>
#define vvl vector<vector<ll>>
#define vvb vector<vector<bool>>
#define vvd vector<vector<double>>
#define vvc vector<vector<char>>
#define vvs vector<vector<string>>
#define vvpi vector<vector<pii>>
#define vvpl vector<vector<pll>>

const int INF = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N;
	string S, T;

	cin >> N >> S >> T;

	int S0Cnt = 0, T0Cnt = 0;
	for (int i = 0; i < N; i++) {		
		if (S[i] == '0') S0Cnt++;
		if (T[i] == '0') T0Cnt++;
	}

	if (S0Cnt != T0Cnt) {
		cout << -1 << '\n';
		return 0;
	}

	int a0 = 0, a1 = 0;
	for (int i = 0; i < N; i++) {
		if (S[i] == T[i]) continue;
		if (S[i] == '0') {
			if (a1) a1--;
			a0++;
		}
		else {
			if (a0) a0--;
			a1++;
		}
	}

	cout << a0 + a1 << '\n';

	return 0;
}