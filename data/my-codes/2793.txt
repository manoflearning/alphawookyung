#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <cassert>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
#define vi vector<int>
#define vvi vector<vector<int>>
#define vl vector<ll>
#define vvl vector<vector<ll>>
#define vs vector<string>
#define vvs vector<vector<string>>
 
const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };
 
int n, a[360];
vi b;
 
void input() {
	cin >> n;
	for (int i = 0; i < n; i++) cin >> a[i];
}
 
int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
 
	input();
 
	int sum = 0;
	b.push_back(0);
	for (int i = 0; i < n; i++) {
		sum += a[i];
		b.push_back(sum % 360);
	}
 
	sort(b.begin(), b.end());
 
	int ans = 0;
	for (int i = 0; i < b.size() - 1; i++) {
		ans = max(ans, b[i + 1] - b[i]);
	}
 
	ans = max(ans, 360 - b[n]);
 
	cout << ans;
 
	return 0;
}
