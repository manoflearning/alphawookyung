#include <iostream>
#include <vector>
#include <set>
using namespace std;

int N, M;
vector<int> arr, ans;

void making(int flag, int digit);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> N >> M;

	set<int> s;
	for (int i = 0; i < N; i++) {
		int x; cin >> x;
		s.insert(x);
	}

	set<int>::iterator it = s.begin();
	while (it != s.end()) {
		arr.push_back(*it);
		it++;
	}
	
	making(0, 0);

	return 0;
}

void making(int flag, int digit) {
	if (digit == M) {
		for (int x : ans) 
			cout << x << ' ';
		cout << '\n';
		return;
	}

	for (int i = flag; i < arr.size(); i++) {
		ans.push_back(arr[i]);
		making(i, digit + 1);
		ans.pop_back();
	}
}