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
const long double PI = 3.141592653589793238462643383279502884197;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	int N, M, K;
	cin >> N >> M >> K;

	vl A(N), B(M), canMake, cnt;

	for (auto& i : A) cin >> i;
	for (auto& i : B) cin >> i;
	
	ll sum = 0;
	for (int i = 0; i < N; i++) {
		sum += A[i];
		if (sum > K) break;
		canMake.push_back(sum);
		cnt.push_back(i + 1);
	}
	
	ll ans = 0;
	
	int loc = lower_bound(canMake.begin(), canMake.end(), K) - canMake.begin();
	
	if (loc == cnt.size() || canMake[loc] > K) {
		if (loc - 1 >= 0) ans = max(ans, cnt[loc - 1]);
	}
	else ans = max(ans, cnt[loc]);
	
	sum = 0;
	for (int i = 0; i < M; i++) {
		sum += B[i];
		if (sum > K) break;
		int loc = lower_bound(canMake.begin(), canMake.end(), K - sum) - canMake.begin();
		
		if (loc == cnt.size() || canMake[loc] > K - sum) {
			if (loc - 1 >= 0) ans = max(ans, i + 1 + cnt[loc - 1]);
			else ans = max<ll>(ans, i + 1);
		}
		else {
			ans = max(ans, i + 1 + cnt[loc]);
		}
	}
	
	cout << ans << '\n';

	return 0;
}