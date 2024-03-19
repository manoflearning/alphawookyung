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

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	string s;
	getline(cin, s);
	
	for (int i = 0; i < s.size(); i++) {
	    stack<char> st;
	    while (i < s.size() && s[i] != ' ' && s[i] != '<') st.push(s[i++]);
	        
	    while (!st.empty()) {
            cout << st.top();
	        st.pop();
	    }
	    
	    if (s[i] == '<') {
	        while (s[i] != '>') cout << s[i++];
	        cout << s[i];
	    }
	    if (s[i] == ' ') cout << s[i];
	}
	
	return 0;
}
