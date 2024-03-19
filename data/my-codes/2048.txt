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

const int MAX = 5e4;

int n, ans, a = 1, b;
vector<int> pp;

bool isPrime[MAX + 1];
vector<int> prime(1, 2);

void getPrime() {
	fill(isPrime, isPrime + MAX + 1, true);

	for (ll i = 4; i <= MAX; i += 2)
		isPrime[i] = false;
	for (ll i = 3; i <= MAX; i++) {
		if (isPrime[i]) prime.push_back(i);
		for (ll j = i * i; j <= MAX; j += i * 2)
			isPrime[j] = false;
	}
}

void input() {
    cin >> n;
}

void f2(int idx, int x) {
    if (b) {
        if (b & 1) ans -= n / a - x / a;
        else ans += n / a - x / a;
    }
    
    for (int i = idx + 1; i < pp.size(); i++) {
        a *= pp[i], b += 1;
        f2(i, x);
        a /= pp[i], b -= 1;
    }
}

void f() {
    for (int i = 1; i <= n; i++) {
        pp.clear();
        
        for (int p : prime) {
            if (i < p) break;
            if (i % p == 0) pp.push_back(p);
        }
        
        ans += n - i;
        f2(-1, i);
    }
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	getPrime();
	
	input();
	
	f();
	
	cout << ans;
	
	return 0;
}
