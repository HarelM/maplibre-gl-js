import fs from 'fs';
import puppeteer from 'puppeteer';
import PDFMerger from 'pdf-merger-js';
import minimist from 'minimist';

const argv = minimist(process.argv.slice(2), {boolean: ['gpu', 'profile']});

const formatTime = (v) => {
    if (typeof v === 'number' && !isNaN(v)) {
        return `${v.toFixed(4)} ms`;
    } else {
        return '';
    }
};

const formatRegression = (v) => {
    if (v) {
        const correlation = v.correlation;
        if (correlation < 0.9) {
            return '\u2620\uFE0F';
        } else if (correlation < 0.99) {
            return '\u26A0\uFE0F';
        }
    }
    return ' ';
};

const dir = './test/bench/results';
if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir);
}

const url = new URL('http://localhost:9966/test/bench/versions/index.html');

if (argv.compare !== true && argv.compare !== undefined) { // handle --compare without argument as the default
    for (const compare of [].concat(argv.compare))
        url.searchParams.append('compare', compare || '');
}

console.log(`Starting headless chrome at: ${url.toString()}`);

const gpuArgs = argv.gpu ? [
    '--no-sandbox',
    '--enable-gpu',
    '--ignore-gpu-blocklist',
    '--disable-software-rasterizer',
    '--enable-features=Vulkan',
    '--use-vulkan',
    '--use-angle=vulkan',
] : [];

const browser = await puppeteer.launch({
    headless: true,
    executablePath: argv.gpu ? '/usr/bin/google-chrome' : undefined,
    args: gpuArgs,
    protocolTimeout: 0,
});

try {

    const webPage = await browser.newPage();
    await webPage.setDefaultTimeout(0);
    await webPage.setViewport({width: 1280, height: 1024});

    url.hash = 'NONE'; // this will simply load the page without running any benchmarks
    await webPage.goto(url.toString());

    await webPage.waitForFunction(() => (window as any).maplibreglBenchmarkFinished);
    const allNames = await webPage.evaluate(() => Object.keys((window as any).maplibreglBenchmarks));
    const versions = await webPage.evaluate((name) => Object.keys((window as any).maplibreglBenchmarks[name]), allNames[0]);
    const versionsDisplayName = await webPage.evaluate(() => (window as any).versionsDisplayName);

    // The following will run all the tests if no arguments are passed, will run only the tests passed as arguments otherwise
    const toRun = argv._.length > 0 ? argv._ : allNames;

    const nameWidth = Math.max(...toRun.map(v => v.length)) + 1;
    const timeWidth = Math.max(...versions.map(v => v.length), 16);

    console.log(''.padStart(nameWidth), ...versions.map((v, i) =>  `${(versionsDisplayName[i]).padStart(timeWidth)} `));

    const merger = new PDFMerger();
    for (const name of toRun) {
        process.stdout.write(name.padStart(nameWidth));

        url.hash = name;
        await webPage.goto(url.toString());
        await webPage.reload();

        await webPage.waitForFunction(
            () => (window as any).maplibreglBenchmarkFinished,
            {
                polling: 200,
                timeout: 0
            }
        );
        const results = await webPage.evaluate((name) => (window as any).maplibreglBenchmarkResults[name], name);
        const output = versions.map((v) => {
            if (v && results[v]) {
                const trimmedMean = results[v].summary?.trimmedMean;
                const regression = results[v].regression;
                return formatTime(trimmedMean).padStart(timeWidth) + formatRegression(regression);
            } else {
                return ''.padStart(timeWidth + 1);
            }
        });
        if (versions.length === 2) {
            const [main, current] = versions;
            const delta = results[current]?.summary?.trimmedMean - results[main]?.summary?.trimmedMean;
            output.push(((delta > 0 ? '+' : '') + formatTime(delta)).padStart(15));
        }
        console.log(...output);

        await merger.add(await webPage.pdf({
            format: 'a4',
            path: `${dir}/${name}.pdf`,
            printBackground: true,
            margin: {
                top: '1cm',
                bottom: '1cm',
                left: '1cm',
                right: '1cm'
            }
        }));
    }

    await merger.save(`${dir}/all.pdf`);

    // CPU profiling mode: re-setup the last benchmark, then profile
    // its render loop using V8's sampling profiler via CDP.
    if (argv.profile && toRun.length > 0) {
        const profileTarget = toRun[toRun.length - 1];
        const profileRenders = 500;
        console.log(`\nProfiling ${profileTarget} (${profileRenders} renders)...`);

        // Navigate to the benchmark and wait for it to finish (which runs setup)
        url.hash = profileTarget;
        await webPage.goto(url.toString());
        await webPage.reload();
        await webPage.waitForFunction(
            () => (window as any).maplibreglBenchmarkFinished,
            {polling: 200, timeout: 0}
        );

        // Re-setup: create a fresh benchmark instance and run its setup
        await webPage.evaluate(async (name) => {
            const benchmarks = (window as any).maplibreglBenchmarks[name];
            const currentVersion = Object.keys(benchmarks).pop();
            const BenchClass = benchmarks[currentVersion].constructor;
            const instance = new BenchClass();
            await instance.setup();
            (window as any)._profileBench = instance;
        }, profileTarget);

        // Wait for everything to settle
        await new Promise(r => setTimeout(r, 3000));

        // Start V8 CPU profiler
        const client = await webPage.createCDPSession();
        await client.send('Profiler.enable');
        await client.send('Profiler.setSamplingInterval', {interval: 10});
        await client.send('Profiler.start');

        // Run bench() in a tight synchronous loop
        await webPage.evaluate((count) => {
            const bench = (window as any)._profileBench;
            for (let i = 0; i < count; i++) {
                bench.bench();
            }
        }, profileRenders);

        const {profile} = await client.send('Profiler.stop');

        // Teardown
        await webPage.evaluate(() => {
            (window as any)._profileBench?.teardown?.();
            delete (window as any)._profileBench;
        });

        // Parse profile: compute self time per function from samples + deltas
        const nodes = {};
        for (const node of profile.nodes) nodes[node.id] = node;

        const selfTime: {[key: string]: number} = {};
        for (let i = 0; i < profile.samples.length; i++) {
            const delta = profile.timeDeltas[i];
            const node = nodes[profile.samples[i]];
            const fn = node.callFrame;
            const fnName = fn.functionName || '(anonymous)';
            const fnUrl = fn.url ? fn.url.split('/').pop() : '';
            const key = fnName + (fnUrl ? ` (${fnUrl}:${fn.lineNumber})` : '');
            selfTime[key] = (selfTime[key] || 0) + delta / 1000;
        }

        const sorted = Object.entries(selfTime)
            .filter(([name]) => name !== '(idle)' && name !== '(program)')
            .sort((a, b) => b[1] - a[1])
            .slice(0, 30);

        const totalTime = (profile.endTime - profile.startTime) / 1000;
        const idleTime = selfTime['(idle)'] || 0;
        const activeTime = totalTime - idleTime;

        console.log(`\n${profileRenders} renders in ${activeTime.toFixed(0)} ms (${(activeTime / profileRenders).toFixed(2)} ms/render)`);
        console.log(`\nTop 30 functions by self time:`);
        console.log(`${'Self(ms)'.padStart(10)} ${'%'.padStart(6)} ${'Per render'.padStart(12)}  Function`);
        for (const [fnName, ms] of sorted) {
            const pct = (ms / activeTime * 100).toFixed(1);
            const perRender = (ms / profileRenders).toFixed(3);
            console.log(`${ms.toFixed(1).padStart(10)} ${pct.padStart(6)} ${(perRender + ' ms').padStart(12)}  ${fnName}`);
        }
    }
} catch (error) {
    if (error.message.startsWith('net::ERR_CONNECTION_REFUSED')) {
        console.log('Could not connect to server. Please run \'npm run start-bench\'.');
    } else {
        console.log(error);
    }
} finally {
    await browser.close();
}
