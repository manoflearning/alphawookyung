#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <cassert>
using namespace std;
#define ll long long
#define fr first
#define sc second
#define pii pair<int, int>
#define pll pair<ll, ll>
#define vi vector<int>
#define vvi vector<vector<int>>
#define vl vector<ll>
#define vvl vector<vector<ll>>
#define vs vector<string>
#define vvs vector<vector<string>>

const ll INF = (ll)1e18 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1, 1, 1, -1, -1 };
const int dx[] = { 1, -1, 0, 0, 1, -1, 1, -1 };

struct ld {
	ll l, d;
};

struct cmp1 {
	bool operator()(ld a, ld b) {
		return a.l + a.d < b.l + b.d;
	}
};

struct cmp2 {
	bool operator()(ld a, ld b) {
		return a.d < b.d;
	}
};

int n;
vector<ld> a;

void input() {
	cin >> n;
	a.resize(n);
	for (auto& i : a) {
		cin >> i.l >> i.d;
	}
}

int f() {
	priority_queue<ld, vector<ld>, cmp2> pq;

	ll sum = 0;
	for (auto& i : a) {
		if (pq.empty()) {
			pq.push(i);
			sum += i.d;
		}
		else {
			if (sum <= i.l) {
				pq.push(i);
				sum += i.d;
			}
			else {
				if (sum - pq.top().d <= i.l && i.d < pq.top().d) {
					sum -= pq.top().d;
					pq.pop();

					sum += i.d;
					pq.push(i);
				}
			}
		}
	}

	return pq.size();
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	input();

	sort(a.begin(), a.end(), cmp1());

	cout << f();

	return 0;
}
