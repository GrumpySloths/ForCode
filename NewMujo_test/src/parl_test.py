import parl
import time


@parl.remote_class(wait=False)
class A(object):

    def run(self):
        ans = 0
        for i in range(100000000):
            ans += i
        return ans


parl.connect("localhost:6111")
start = time.time()

actors = [A() for _ in range(5)]
jobs = [actor.run() for actor in actors]
returns = [job.get() for job in jobs]
true_result = sum([i for i in range(100000000)])
for result in returns:
    assert result == true_result
end = time.time()
print("time:", end - start)
