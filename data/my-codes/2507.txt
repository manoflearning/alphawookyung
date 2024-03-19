#include <iostream>
#define ll long long
using namespace std;

int arr1[100], arr2[100];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N; cin >> N;

	for (int i = 0; i < N; i++)
		cin >> arr1[i];
	for (int i = 0; i < N; i++)
		cin >> arr2[i];
	
	int more = 0, less = 0;
	for (int i = 0; i < N; i++) {
		if (arr1[i] == 0 && arr2[i] == 0) continue;
		else if (arr1[i] == 1 && arr2[i] == 0) more++;
		else if (arr1[i] == 0 && arr2[i] == 1) less++;
	}

	if (more == 0) cout << "-1";
	else {
		for (int i = 1; ; i++) {
			if (i * more > less) {
				cout << i;
				break;
			}
		}
	}

	return 0;
}