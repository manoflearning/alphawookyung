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

	int tc; cin >> tc;

	while (tc--) {
		int N; cin >> N;
		
		vi odd, even;
		for (int i = 1; i <= 2 * N; i++) {
			int x; cin >> x;
			if (x & 1) odd.push_back(i);
			else even.push_back(i);
		}

		if (odd.size() & 1) {
			odd.pop_back();
			even.pop_back();
		}
		else {
			if (odd.size() >= 2) {
				odd.pop_back();
				odd.pop_back();
			}
			else {
				even.pop_back();
				even.pop_back();
			}
		}

		while (!odd.empty()) {
			cout << odd.back() << ' ';
			odd.pop_back();
			cout << odd.back() << '\n';
			odd.pop_back();
		}

		while (!even.empty()) {
			cout << even.back() << ' ';
			even.pop_back();
			cout << even.back() << '\n';
			even.pop_back();
		}
	}

	return 0;
}