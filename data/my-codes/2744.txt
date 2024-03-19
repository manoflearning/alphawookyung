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
const int MAX = 1e7;

bool isPrime[MAX + 1];
vector<int> prime(1, 2);

void getPrime() {
    fill(isPrime, isPrime + MAX + 1, true);

    for (ll i = 4; i <= MAX; i += 2)
        isPrime[i] = false;
    for (ll i = 3; i <= MAX; i++) {
        if (isPrime[i])
            prime.push_back(i);
	for (ll j = i * i; j <= MAX; j += i * 2)
	    isPrime[j] = false;
    }
}

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);
    
    getPrime();
    
    ll n=600851475143, res;
    for(ll i : prime){
        if(n%i==0) res=i;
    }
    
    cout<<res;
    
    return 0;
}
