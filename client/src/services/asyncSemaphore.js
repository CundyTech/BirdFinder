// A simple concurrency limiter. Used to stop the life-list grid (60 tiles,
// each firing a species lookup on mount) from launching dozens of
// iNaturalist requests at once — that burst was tripping timeouts/rate
// limits and made every lookup fail together, not just the excess ones.
export function createSemaphore(maxConcurrent) {
  let active = 0;
  const queue = [];

  function runNext() {
    if (active >= maxConcurrent || queue.length === 0) return;
    active += 1;
    const { task, resolve, reject } = queue.shift();
    task()
      .then(resolve, reject)
      .finally(() => {
        active -= 1;
        runNext();
      });
  }

  return function schedule(task) {
    return new Promise((resolve, reject) => {
      queue.push({ task, resolve, reject });
      runNext();
    });
  };
}
