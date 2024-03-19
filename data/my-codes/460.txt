#include <iostream>
#include <algorithm>
using namespace std;

int N;

int f(int num);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> N;
	cout << f(0);

	return 0;
}

int f(int num) {
	if (num > N) return num / 10;
	return max(f(num * 10 + 4), f(num * 10 + 7));
}