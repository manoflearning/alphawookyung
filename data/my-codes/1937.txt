#include <iostream>
#include <algorithm>
using namespace std;

int arr[55];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	int K; cin >> K;

	for (int test = 1; test <= K; test++) {
		int N; cin >> N;
		for (int i = 1; i <= N; i++)
			cin >> arr[i];

		sort(arr + 1, arr + N + 1);

		int gap = -1;
		for (int i = N; i >= 2; i--)
			gap = max(gap, arr[i] - arr[i - 1]);

		cout << "Class " << test << '\n';
		cout << "Max " << arr[N] << ", ";
		cout << "Min " << arr[1] << ", ";
		cout << "Largest gap " << gap << '\n';
	}

	return 0;
}