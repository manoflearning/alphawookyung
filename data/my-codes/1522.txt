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
const int MAX = 1e6;

bool isPrime[MAX + 1];
vector<int> prime;

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

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);
    
    getPrime();

    while(true) {
        int x; cin>>x;
        if(x==0) break;

        bool yes = 0;
        for(int i=0; i<prime.size() && prime[i]<=x-prime[i]; i++) {
            if(isPrime[x-prime[i]]) {
                cout<<x<<" = "<<prime[i]<<" + "<<x-prime[i]<<'\n';
                yes = 1;
                break;
            }
        }
        if(!yes) cout<<"Goldbach's conjecture is wrong.\n";
    }

    return 0;
}