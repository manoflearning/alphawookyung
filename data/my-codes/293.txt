#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <cassert>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

struct MV {
	int M, V;
};

bool operator<(const MV& a, const MV& b) {
	return a.V < b.V;
}

int N, K;
vector<MV> jew;
vector<int> bag;
priority_queue<MV> pq;

bool cmp(MV& a, MV& b) {
	return a.M < b.M;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	cout << fixed; cout.precision(6);

	cin >> N >> K;
	jew.resize(N), bag.resize(K);
	for (auto& i : jew) cin >> i.M >> i.V;
	for (auto& i : bag) cin >> i;

	sort(jew.begin(), jew.end(), cmp);
	sort(bag.begin(), bag.end());

	int l = 0;
	ll res = 0;
	for (int i = 0; i < bag.size(); i++) {
		while (l < N && jew[l].M <= bag[i]) pq.push(jew[l++]);
		
		if (!pq.empty()) {
			res += pq.top().V;
			pq.pop();
		}
	}

	cout << res;

	return 0;
}