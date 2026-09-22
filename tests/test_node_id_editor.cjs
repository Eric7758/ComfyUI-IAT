const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const app = {
    registerExtension(extension) {
        this.extension = extension;
    },
};
const context = vm.createContext({ app, console, structuredClone, window: { LiteGraph: {} } });
const source = fs.readFileSync("js/node_id_editor.js", "utf8").replace(/^import .*;\r?\n/, "");
vm.runInContext(source, context);

function evaluate(sourceText) {
    return vm.runInContext(sourceText, context);
}

function equal(actual, expected) {
    assert.equal(JSON.stringify(actual), JSON.stringify(expected));
}

evaluate(`
function fixture() {
    let data = {
        last_node_id: 9,
        nodes: [
            { id: 9, pos: [0, 0], size: [100, 100] },
            { id: 4, pos: [300, 0], size: [100, 100] },
        ],
        links: [[1, 9, 0, 4, 0, "IMAGE"]],
        floatingLinks: [],
    };
    const graph = { _nodes: [], serialize() { return structuredClone(data); } };
    graph.rootGraph = graph;
    graph._nodes = data.nodes.map((item) => {
        const state = { id: item.id };
        const node = { graph, pos: item.pos, size: item.size };
        Object.defineProperty(node, "id", {
            get() { return state.id; },
            set() {},
        });
        return node;
    });
    app.rootGraph = app.graph = graph;
    app.canvas = { graph, setGraph(nextGraph) { this.graph = nextGraph; } };
    app.loadGraphData = async (workflow) => {
        data = structuredClone(workflow);
        const loaded = {
            _nodes: data.nodes.map((item) => ({ id: item.id, pos: item.pos, size: item.size })),
            serialize() { return structuredClone(data); },
        };
        loaded.rootGraph = loaded;
        for (const node of loaded._nodes) node.graph = loaded;
        app.rootGraph = app.graph = loaded;
        return true;
    };
    return { graph, get data() { return data; } };
}
`);

async function main() {
    let fixture = evaluate("fixture()");
    assert.equal(await evaluate("setNodeId(app.graph._nodes[0], 7)"), true);
    assert.equal(fixture.graph._nodes[0].id, 9, "the simulated current frontend must reject direct ID writes");
    equal(fixture.data.nodes.map((node) => node.id), [7, 4]);
    equal(fixture.data.links[0].slice(1, 4), [7, 0, 4]);
    assert.equal(fixture.data.last_node_id, 7);
    assert.equal(app.canvas.graph, app.rootGraph);

    fixture = evaluate("fixture()");
    await assert.rejects(evaluate("setNodeId(app.graph._nodes[0], 4)"), /already in use/);
    equal(fixture.data.nodes.map((node) => node.id), [9, 4]);

    fixture = evaluate("fixture()");
    equal(await evaluate("resetGraphNodeIds(app.graph)"), { changed: 2, total: 2 });
    equal(fixture.data.nodes.map((node) => node.id), [1, 2]);
    equal(fixture.data.links[0].slice(1, 4), [1, 0, 2]);

    fixture = evaluate("fixture()");
    await evaluate(`remapNodeIds(app.graph, [
        { node: app.graph._nodes[0], nextId: 4 },
        { node: app.graph._nodes[1], nextId: 9 },
    ])`);
    equal(fixture.data.nodes.map((node) => node.id), [4, 9]);
    equal(fixture.data.links[0].slice(1, 4), [4, 0, 9]);

    for (const value of ["0", "-1", "1.5", "abc", "9007199254740992"]) {
        assert.throws(() => evaluate(`parseNodeId(${JSON.stringify(value)})`));
    }
}

main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
