import { app } from "../../scripts/app.js";

const TOAST_SUMMARY = "IAT Node ID";
const NODE_ID_PATTERN = /^[1-9]\d*$/;

function showToast(severity, detail) {
    app.extensionManager?.toast?.add?.({
        severity,
        summary: TOAST_SUMMARY,
        detail,
        life: severity === "error" ? 8000 : 3500,
    });
}

function getGraphNodes(graph) {
    return Array.isArray(graph?._nodes) ? graph._nodes.filter(Boolean) : [];
}

function nodeIdKey(id) {
    return `${id}`;
}

function toRuntimeNodeId(id, previousId) {
    return typeof previousId === "number" ? id : `${id}`;
}

function getCollectionValues(collection) {
    if (!collection) {
        return [];
    }
    if (collection instanceof Map) {
        return collection.values();
    }
    if (Array.isArray(collection)) {
        return collection;
    }
    return Object.values(collection);
}

function getNodeLayout(node) {
    const [x = 0, y = 0] = node?.pos || [];
    const [width = 0, height = 0] = node?.size || [];
    return {
        x,
        y,
        centerX: x + width / 2,
        centerY: y + height / 2,
        width,
        height,
    };
}

function getColumnTolerance(nodes) {
    const widths = nodes
        .map((node) => getNodeLayout(node).width)
        .filter((width) => Number.isFinite(width) && width > 0)
        .sort((left, right) => left - right);

    const medianWidth = widths.length ? widths[Math.floor(widths.length / 2)] : 160;
    return Math.max(48, Math.min(192, medianWidth * 0.35));
}

function compareNodesByPosition(leftNode, rightNode, originalOrder) {
    const left = getNodeLayout(leftNode);
    const right = getNodeLayout(rightNode);

    return left.y - right.y
        || left.x - right.x
        || originalOrder.get(leftNode) - originalOrder.get(rightNode);
}

function sortNodesForReset(nodes) {
    const columnTolerance = getColumnTolerance(nodes);
    const columns = [];
    const originalOrder = new Map(nodes.map((node, index) => [node, index]));

    const nodesByX = [...nodes].sort((leftNode, rightNode) => {
        const left = getNodeLayout(leftNode);
        const right = getNodeLayout(rightNode);
        return left.centerX - right.centerX
            || left.y - right.y
            || originalOrder.get(leftNode) - originalOrder.get(rightNode);
    });

    for (const node of nodesByX) {
        const layout = getNodeLayout(node);
        const column = columns.at(-1);

        if (!column || Math.abs(layout.centerX - column.centerX) > columnTolerance) {
            columns.push({
                centerX: layout.centerX,
                nodes: [node],
            });
            continue;
        }

        column.nodes.push(node);
        column.centerX = (column.centerX * (column.nodes.length - 1) + layout.centerX) / column.nodes.length;
    }

    return columns.flatMap((column) => column.nodes.sort(
        (leftNode, rightNode) => compareNodesByPosition(leftNode, rightNode, originalOrder),
    ));
}

const LINK_NODE_ID_FIELDS = ["origin_id", "target_id", "originId", "targetId"];

function remapLinkEndpoints(links, idMap) {
    for (const link of links) {
        if (Array.isArray(link)) {
            const originId = idMap.get(nodeIdKey(link[1]));
            const targetId = idMap.get(nodeIdKey(link[3]));
            if (originId !== undefined) {
                link[1] = originId;
            }
            if (targetId !== undefined) {
                link[3] = targetId;
            }
            continue;
        }

        for (const field of LINK_NODE_ID_FIELDS) {
            if (!(field in link)) {
                continue;
            }
            const nextId = idMap.get(nodeIdKey(link[field]));
            if (nextId !== undefined) {
                link[field] = nextId;
            }
        }
    }
}

function parseNumericRuntimeNodeId(id) {
    const text = nodeIdKey(id);
    if (!/^(0|[1-9]\d*)$/.test(text)) {
        return null;
    }
    const number = Number(text);
    return Number.isSafeInteger(number) && `${number}` === text ? number : null;
}

function updateSerializedLastNodeId(graphData) {
    let lastNodeId = 0;
    for (const node of graphData.nodes || []) {
        const numericId = parseNumericRuntimeNodeId(node.id);
        if (numericId !== null) {
            lastNodeId = Math.max(lastNodeId, numericId);
        }
    }
    if (graphData.state && "lastNodeId" in graphData.state) {
        graphData.state.lastNodeId = lastNodeId;
    }
    if ("last_node_id" in graphData) {
        graphData.last_node_id = lastNodeId;
    }
}

function findSerializedGraph(workflow, graph) {
    const rootGraph = graph?.rootGraph || graph;
    if (graph === rootGraph) {
        return workflow;
    }

    const pending = [...getCollectionValues(workflow?.definitions?.subgraphs)];
    while (pending.length) {
        const candidate = pending.shift();
        if (nodeIdKey(candidate?.id) === nodeIdKey(graph?.id)) {
            return candidate;
        }
        pending.push(...getCollectionValues(candidate?.definitions?.subgraphs));
    }
    throw new Error("The active subgraph could not be found in the serialized workflow.");
}

function remapSerializedGraph(graphData, changes) {
    const idMap = new Map(changes.map(({ previousId, nextId }) => [nodeIdKey(previousId), nextId]));
    for (const node of graphData.nodes || []) {
        const nextId = idMap.get(nodeIdKey(node.id));
        if (nextId !== undefined) {
            node.id = nextId;
        }
    }
    for (const links of [graphData.links, graphData.floatingLinks]) {
        remapLinkEndpoints(getCollectionValues(links), idMap);
    }
    for (const field of ["inputNode", "outputNode"]) {
        const nextId = idMap.get(nodeIdKey(graphData[field]));
        if (nextId !== undefined) {
            graphData[field] = nextId;
        }
    }
    updateSerializedLastNodeId(graphData);
}

async function reloadRemappedWorkflow(graph, changes) {
    const rootGraph = graph?.rootGraph || graph;
    if (typeof rootGraph?.serialize !== "function" || typeof app.loadGraphData !== "function") {
        throw new Error("This ComfyUI version cannot safely reload a remapped workflow.");
    }

    const workflow = structuredClone(rootGraph.serialize());
    remapSerializedGraph(findSerializedGraph(workflow, graph), changes);
    const loaded = await app.loadGraphData(workflow, false, false);
    if (loaded === false) {
        throw new Error("ComfyUI rejected the remapped workflow.");
    }

    const reloadedRoot = app.rootGraph || app.graph;
    const reloadedGraph = graph === rootGraph
        ? reloadedRoot
        : [...getCollectionValues(reloadedRoot?.subgraphs ?? reloadedRoot?._subgraphs)]
            .find((candidate) => nodeIdKey(candidate?.id) === nodeIdKey(graph.id));
    if (!reloadedGraph) {
        throw new Error("ComfyUI reloaded the workflow without the active graph.");
    }
    for (const { nextId } of changes) {
        if (!getGraphNodes(reloadedGraph).some((node) => nodeIdKey(node.id) === nodeIdKey(nextId))) {
            throw new Error(`ComfyUI reloaded the workflow without node ID ${nextId}.`);
        }
    }
    if (app.canvas?.graph !== reloadedGraph) {
        app.canvas?.setGraph?.(reloadedGraph);
    }
}

function validateRemap(graph, assignments) {
    if (!graph) {
        throw new Error("The selected node is not attached to a graph.");
    }

    const plannedNodes = new Set();
    const targetIds = new Map();
    const planned = assignments.map(({ node, nextId }) => {
        if (node?.graph !== graph || node.id === null || node.id === undefined) {
            throw new Error("The selected node is not attached to this graph.");
        }
        if (!Number.isSafeInteger(nextId) || nextId < 1) {
            throw new Error("Node ID must be a positive integer.");
        }

        const runtimeId = toRuntimeNodeId(nextId, node.id);
        const targetKey = nodeIdKey(runtimeId);
        if (targetIds.has(targetKey) && targetIds.get(targetKey) !== node) {
            throw new Error(`Node ID ${targetKey} is assigned more than once.`);
        }
        targetIds.set(targetKey, node);
        plannedNodes.add(node);
        return { node, previousId: node.id, nextId: runtimeId };
    });

    for (const node of getGraphNodes(graph)) {
        if (!plannedNodes.has(node) && targetIds.has(nodeIdKey(node.id))) {
            throw new Error(`Node ID ${node.id} is already in use.`);
        }
    }

    return planned.filter(({ previousId, nextId }) => nodeIdKey(previousId) !== nodeIdKey(nextId));
}

async function remapNodeIds(graph, assignments) {
    const changes = validateRemap(graph, assignments);
    if (!changes.length) {
        return 0;
    }
    await reloadRemappedWorkflow(graph, changes);
    return changes.length;
}

async function resetGraphNodeIds(graph) {
    const nodes = getGraphNodes(graph);
    if (!nodes.length) {
        return {
            changed: 0,
            total: 0,
        };
    }

    const orderedNodes = sortNodesForReset(nodes);
    const changed = await remapNodeIds(
        graph,
        orderedNodes.map((node, index) => ({ node, nextId: index + 1 })),
    );
    return {
        changed,
        total: orderedNodes.length,
    };
}

async function setNodeId(node, nextId) {
    const graph = node?.graph;
    return await remapNodeIds(graph, [{ node, nextId }]) > 0;
}

function promptForNodeId(node) {
    const title = node?.title || node?.type || node?.comfyClass || "Node";
    return window.prompt(`Set a new ID for "${title}"`, `${node?.id ?? ""}`);
}

function parseNodeId(value) {
    const trimmed = `${value}`.trim();
    if (!NODE_ID_PATTERN.test(trimmed)) {
        throw new Error("Node ID must be a positive integer.");
    }
    const nodeId = Number.parseInt(trimmed, 10);
    if (!Number.isSafeInteger(nodeId)) {
        throw new Error("Node ID is too large.");
    }
    return nodeId;
}

function confirmResetGraphNodeIds(totalNodes) {
    return window.confirm(
        `Reset ${totalNodes} node IDs in the current graph from 1 using left-to-right priority, then top-to-bottom order?`,
    );
}

app.registerExtension({
    name: "comfyui_iat.node_id_editor",

    getCanvasMenuItems(canvas) {
        const graph = canvas?.graph;
        const totalNodes = getGraphNodes(graph).length;
        if (!graph || !totalNodes) {
            return [];
        }

        return [
            {
                content: "Reset graph node IDs...",
                callback: async () => {
                    if (!confirmResetGraphNodeIds(totalNodes)) {
                        return;
                    }

                    try {
                        const result = await resetGraphNodeIds(graph);
                        if (!result.changed) {
                            showToast("info", `Current graph already uses IDs 1-${result.total} in the default order.`);
                            return;
                        }
                        showToast("success", `Reassigned ${result.changed} node IDs in ${result.total} nodes.`);
                    } catch (error) {
                        const detail = error?.message || "Failed to reset graph node IDs.";
                        showToast("error", detail);
                        console.error("[IAT] failed to reset graph node ids", error);
                    }
                },
            },
        ];
    },

    getNodeMenuItems(node) {
        if (!node?.graph) {
            return [];
        }

        return [
            {
                content: "Set node ID...",
                callback: async () => {
                    const rawValue = promptForNodeId(node);
                    if (rawValue === null) {
                        return;
                    }

                    const previousId = node.id;
                    try {
                        const nextId = parseNodeId(rawValue);
                        const updated = await setNodeId(node, nextId);
                        if (!updated) {
                            showToast("info", `Node already uses ID ${nextId}.`);
                            return;
                        }
                        showToast("success", `Updated node ID from ${previousId} to ${nextId}.`);
                    } catch (error) {
                        const detail = error?.message || "Failed to update node ID.";
                        showToast("error", detail);
                        console.error("[IAT] failed to update node id", error);
                    }
                },
            },
        ];
    },
});
