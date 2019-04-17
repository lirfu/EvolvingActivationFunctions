import datetime


class Stopwatch:
    def start(self):
        self.start_time = self.last_lap = datetime.datetime()

    def lap(self):
        t = datetime.datetime()
        delta = t - self.last_lap
        self.last_lap = t
        return delta

    def stop(self):
        return datetime.datetime() - self.start_time

def roulette_wheel_select(self, array, weights, samples=1):
    suma = sum(weights)
    smp = random.uniform(0, suma, samples)
    results = []
    for x in range(samples):
        acc = 0
        for i in range(len(weights)):
            acc += weights[i]
            if acc >= smp[x]:
                results.append(array[i])
    return results
