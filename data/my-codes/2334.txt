#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

ll R, G, B;
vector<ll> rGem, gGem, bGem;

void init();
void scan();
inline ll cal(ll a, ll b, ll c);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	int T; cin >> T;

	while (T--) {
		init();

		scan();

		sort(rGem.begin(), rGem.end());
		sort(gGem.begin(), gGem.end());
		sort(bGem.begin(), bGem.end());

		ll ans = LLONG_MAX;

		for(int r = 0; r < R; r++) {
			int gR = lower_bound(gGem.begin(), gGem.end(), rGem[r]) - gGem.begin();
			int bR = lower_bound(bGem.begin(), bGem.end(), rGem[r]) - bGem.begin();
			int gL = gR - 1, bL = bR - 1;

			if (gR < G && bR < B) {
				if (gGem[gR] < bGem[bR]) {
					ll mid = (rGem[r] + bGem[bR]) / 2;
					int g = lower_bound(gGem.begin(), gGem.end(), mid) - gGem.begin();
					if (G > g && g >= 0) ans = min(ans, cal(rGem[r], gGem[g], bGem[bR]));
					g--;
					if (G > g && g >= 0) ans = min(ans, cal(rGem[r], gGem[g], bGem[bR]));
				}
				else {
					ll mid = (rGem[r] + gGem[gR]) / 2;
					int b = lower_bound(bGem.begin(), bGem.end(), mid) - bGem.begin();
					if (B > b && b >= 0) ans = min(ans, cal(rGem[r], gGem[gR], bGem[b]));
					b--;
					if (B > b && b >= 0) ans = min(ans, cal(rGem[r], gGem[gR], bGem[b]));
				}
			}
			if (0 <= gL && bR < B) {
				int mid = (gGem[gL] + bGem[bR]) / 2;
				int rr = lower_bound(rGem.begin(), rGem.end(), mid) - rGem.begin();
				if (R > rr && rr >= 0) ans = min(ans, cal(rGem[rr], gGem[gL], bGem[bR]));
				rr--;
				if (R > rr && rr >= 0) ans = min(ans, cal(rGem[rr], gGem[gL], bGem[bR]));
			}
			if (0 <= bL && gR < G) {
				int mid = (bGem[bL] + gGem[gR]) / 2;
				int rr = lower_bound(rGem.begin(), rGem.end(), mid) - rGem.begin();
				if (R > rr && rr >= 0) ans = min(ans, cal(rGem[rr], gGem[gR], bGem[bL]));
				rr--;
				if (R > rr && rr >= 0) ans = min(ans, cal(rGem[rr], gGem[gR], bGem[bL]));
			}
			if (0 <= gL && 0 <= bL) {
				if (gGem[gL] < bGem[bL]) {
					int mid = (gGem[gL] + rGem[r]) / 2;
					int b = lower_bound(bGem.begin(), bGem.end(), mid) - bGem.begin();
					if (B > b && b >= 0) ans = min(ans, cal(rGem[r], gGem[gL], bGem[b]));
					b--;
					if (B > b && b >= 0) ans = min(ans, cal(rGem[r], gGem[gL], bGem[b]));
				}
				else {
					int mid = (bGem[bL] + rGem[r]) / 2;
					int g = lower_bound(gGem.begin(), gGem.end(), mid) - gGem.begin();
					if (G > g && g >= 0) ans = min(ans, cal(rGem[r], gGem[g], bGem[bL]));
					g--;
					if (G > g && g >= 0) ans = min(ans, cal(rGem[r], gGem[g], bGem[bL]));
				}
			}
		}

		cout << ans << '\n';
	}

	return 0;
}

void init() {
	rGem.clear();
	gGem.clear(); 
	bGem.clear();
}

void scan() {
	cin >> R >> G >> B;

	for (int i = 0; i < R; i++) {
		ll x; cin >> x;
		rGem.push_back(x);
	}

	for (int i = 0; i < G; i++) {
		ll x; cin >> x;
		gGem.push_back(x);
	}

	for (int i = 0; i < B; i++) {
		ll x; cin >> x;
		bGem.push_back(x);
	}
}

inline ll cal(ll a, ll b, ll c) {
	return (a - b) * (a - b) + (b - c) * (b - c) + (c - a) * (c - a);
}