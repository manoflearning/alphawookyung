#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int N, M, label[10005], lCnt;
bool visited[8];
vector<int> arr, ans;

void making(int digit);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> N >> M;

	arr.resize(N);
	for (int& x : arr) {
		cin >> x;
		if (label[x] == 0) label[x] = ++lCnt;
	}

	sort(arr.begin(), arr.end());

	making(0);

	return 0;
}

void making(int digit) {
	if (digit == M) {
		for (int x : ans) 
			cout << x << ' ';
		cout << '\n';
		return;
	}

	vector<bool> visited2(9);

	for (int i = 0; i < N; i++) {
		if (visited[i] || visited2[label[arr[i]]]) continue;
		
		visited2[label[arr[i]]] = true;

		ans.push_back(arr[i]);
		visited[i] = true;
		
		making(digit + 1);

		ans.pop_back();
		visited[i] = false;
	}
}