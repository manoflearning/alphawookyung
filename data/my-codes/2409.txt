#include <iostream>
#include <vector>
#include <algorithm>
#define ll long long
using namespace std;

const int MAX = 1e5;

int N, arr[MAX + 5];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		cin >> N;

		for (int i = 0; i < N; i++)
			cin >> arr[i];

		sort(arr, arr + N);

		int cnt = N;

		for (int i = 1; i < N; i++) {
			if (arr[i] == arr[i - 1]) cnt--;
		}

		cout << cnt << '\n';
	}

	return 0;
}