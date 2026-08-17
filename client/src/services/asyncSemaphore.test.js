import { createSemaphore } from './asyncSemaphore';

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

describe('createSemaphore', () => {
  it('never runs more than maxConcurrent tasks at once', async () => {
    const schedule = createSemaphore(2);
    const gates = [deferred(), deferred(), deferred()];
    let active = 0;
    let maxActive = 0;

    const results = gates.map((gate, i) =>
      schedule(async () => {
        active += 1;
        maxActive = Math.max(maxActive, active);
        await gate.promise;
        active -= 1;
        return i;
      })
    );

    // The third task shouldn't have started yet — only 2 slots exist.
    expect(active).toBe(2);
    expect(maxActive).toBe(2);

    gates[0].resolve();
    gates[1].resolve();
    gates[2].resolve();

    expect(await Promise.all(results)).toEqual([0, 1, 2]);
    expect(maxActive).toBe(2);
  });

  it('runs tasks immediately while under the limit', async () => {
    const schedule = createSemaphore(4);
    const result = await schedule(async () => 'done');
    expect(result).toBe('done');
  });

  it('propagates a rejected task to its own caller without blocking the queue', async () => {
    const schedule = createSemaphore(1);
    const first = schedule(async () => {
      throw new Error('boom');
    });
    const second = schedule(async () => 'ok');

    await expect(first).rejects.toThrow('boom');
    await expect(second).resolves.toBe('ok');
  });

  it('starts a queued task once a running one finishes', async () => {
    const schedule = createSemaphore(1);
    const gate = deferred();
    let secondStarted = false;

    const firstDone = schedule(async () => {
      await gate.promise;
    });
    const secondDone = schedule(async () => {
      secondStarted = true;
    });

    // Still holding the single slot — the second task can't have started.
    expect(secondStarted).toBe(false);

    gate.resolve();
    await firstDone;
    await secondDone;

    expect(secondStarted).toBe(true);
  });
});
