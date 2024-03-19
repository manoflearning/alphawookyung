#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

int main() {
	int t;
	scanf("%d", &t);

	for (int test = 0; test < t; test++) {
		int n;
		scanf("%d", &n);

		vector<int> ans;

		int flag;
		for (flag = 0; pow(2, flag) <= n; flag++); flag--;

		for (; flag >= 0; flag--) {
			int check = pow(2, flag);

			if (check <= n) {
				n -= check;
				ans.push_back(flag);
			}
		}

		for (int i = ans.size() - 1; i >= 0; i--)
			printf("%d ", ans[i]);
	}

	return 0;
}