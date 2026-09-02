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

function getMutableGraphLinks(graph) {
    const links = [];
    const seen = new Set();
    const collections = [
        graph?._links ?? graph?.links,
        graph?.floatingLinksInternal ?? graph?.floatingLinks,
    ];

    for (const collection of collections) {
        for (const link of getCollectionValues(collection)) {
            if (!link || seen.has(link)) {
                continue;
            }
            seen.add(link);
            links.push(link);
        }
    }
    return links;
}

function getWorkflowGraphs(graph) {
    const rootGraph = graph?.rootGraph || graph;
    const graphs = new Set([rootGraph, graph].filter(Boolean));
    const subgraphs = rootGraph?.subgraphs ?? rootGraph?._subgraphs;

    for (const subgraph of getCollectionValues(subgraphs)) {
        if (subgraph) {
            graphs.add(subgraph);
        }
    }
    return [...graphs];
}

function ensureRemapSupported() {
    if (window.LiteGraph?.vueNodesMode) {
        throw new Error(
            "Node ID editing is unavailable while Nodes 2.0 is enabled. Switch to the LiteGraph renderer and try again.",
        );
    }
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

function captureLinkEndpoints(links) {
    return links.map((link) => {
        if (Array.isArray(link)) {
            return { link, values: [link[1], link[3]] };
        }

        const values = {};
        for (const field of LINK_NODE_ID_FIELDS) {
            if (field in link) {
                values[field] = link[field];
            }
        }
        return { link, values };
    });
}

function restoreLinkEndpoints(snapshots) {
    for (const { link, values } of snapshots) {
        if (Array.isArray(link)) {
            [link[1], link[3]] = values;
            continue;
        }
        for (const [field, value] of Object.entries(values)) {
            link[field] = value;
        }
    }
}

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

function rebuildNodeIndex(graph) {
    const nodes = getGraphNodes(graph);
    const index = graph?._nodes_by_id;

    if (index instanceof Map) {
        index.clear();
        for (const node of nodes) {
            index.set(node.id, node);
        }
        return;
    }

    const nextIndex = Object.fromEntries(nodes.map((node) => [node.id, node]));
    if (!index || typeof index !== "object") {
        graph._nodes_by_id = nextIndex;
        return;
    }

    for (const key of Object.keys(index)) {
        delete index[key];
    }
    Object.assign(index, nextIndex);
}

function getGraphCanvases(graph) {
    return new Set([
        app.canvas?.graph === graph ? app.canvas : null,
        ...(graph?.list_of_graphcanvas || []),
    ].filter(Boolean));
}

function rebuildCanvasSelectionIndexes(graph) {
    const graphNodes = new Set(getGraphNodes(graph));

    for (const canvas of getGraphCanvases(graph)) {
        const selectedNodes = new Set();
        const selectedIndexValues = canvas.selected_nodes instanceof Map
            ? canvas.selected_nodes.values()
            : Object.values(canvas.selected_nodes || {});
        for (const node of selectedIndexValues) {
            if (graphNodes.has(node)) {
                selectedNodes.add(node);
            }
        }
        if (canvas.selectedItems instanceof Set) {
            for (const item of canvas.selectedItems) {
                if (graphNodes.has(item)) {
                    selectedNodes.add(item);
                }
            }
        }

        if (canvas.selected_nodes instanceof Map) {
            canvas.selected_nodes.clear();
            for (const node of selectedNodes) {
                canvas.selected_nodes.set(node.id, node);
            }
            continue;
        }

        const index = canvas.selected_nodes || {};
        for (const key of Object.keys(index)) {
            delete index[key];
        }
        for (const node of selectedNodes) {
            index[node.id] = node;
        }
        canvas.selected_nodes = index;
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

function getLastNodeIdSnapshot(graph) {
    const rootGraph = graph?.rootGraph || graph;
    if (rootGraph?.state && "lastNodeId" in rootGraph.state) {
        return { owner: rootGraph.state, property: "lastNodeId", value: rootGraph.state.lastNodeId };
    }
    if (rootGraph && "last_node_id" in rootGraph) {
        return { owner: rootGraph, property: "last_node_id", value: rootGraph.last_node_id };
    }
    return null;
}

function updateWorkflowLastNodeId(graph) {
    const snapshot = getLastNodeIdSnapshot(graph);
    if (!snapshot) {
        return;
    }

    let lastNodeId = 0;
    for (const workflowGraph of getWorkflowGraphs(graph)) {
        for (const node of getGraphNodes(workflowGraph)) {
            const numericId = parseNumericRuntimeNodeId(node.id);
            if (numericId !== null) {
                lastNodeId = Math.max(lastNodeId, numericId);
            }
        }
    }
    snapshot.owner[snapshot.property] = lastNodeId;
}

function rebindNodeWidgets(nodes) {
    for (const node of nodes) {
        try {
            for (const widget of node.widgets || []) {
                widget?.setNodeId?.(node.id);
            }
        } catch (error) {
            console.warn(`[IAT] failed to rebind widgets for node ${node.id}`, error);
        }
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

    for (const workflowGraph of getWorkflowGraphs(graph)) {
        for (const node of getGraphNodes(workflowGraph)) {
            if (!plannedNodes.has(node) && targetIds.has(nodeIdKey(node.id))) {
                throw new Error(`Node ID ${node.id} is already in use.`);
            }
        }
    }

    return planned.filter(({ previousId, nextId }) => nodeIdKey(previousId) !== nodeIdKey(nextId));
}

function remapNodeIds(graph, assignments) {
    ensureRemapSupported();
    const changes = validateRemap(graph, assignments);
    if (!changes.length) {
        return 0;
    }

    const nodes = getGraphNodes(graph);
    const nodeSnapshots = changes.map(({ node, previousId }) => ({ node, previousId }));
    const links = getMutableGraphLinks(graph);
    const linkSnapshots = captureLinkEndpoints(links);
    const lastNodeIdSnapshot = getLastNodeIdSnapshot(graph);
    const idMap = new Map(changes.map(({ previousId, nextId }) => [nodeIdKey(previousId), nextId]));

    graph.beforeChange?.();
    try {
        for (const { node, nextId } of changes) {
            node.id = nextId;
        }
        rebuildNodeIndex(graph);
        remapLinkEndpoints(links, idMap);
        rebuildCanvasSelectionIndexes(graph);
        updateWorkflowLastNodeId(graph);
    } catch (error) {
        for (const { node, previousId } of nodeSnapshots) {
            node.id = previousId;
        }
        restoreLinkEndpoints(linkSnapshots);
        rebuildNodeIndex(graph);
        rebuildCanvasSelectionIndexes(graph);
        if (lastNodeIdSnapshot) {
            lastNodeIdSnapshot.owner[lastNodeIdSnapshot.property] = lastNodeIdSnapshot.value;
        }
        throw error;
    } finally {
        graph.afterChange?.();
    }

    const changedNodes = new Set(changes.map(({ node }) => node));
    rebindNodeWidgets(changedNodes);
    graph.change?.();
    for (const canvas of getGraphCanvases(graph)) {
        canvas.setDirty?.(true, true);
    }
    for (const node of nodes) {
        if (changedNodes.has(node)) {
            node.setDirtyCanvas?.(true, true);
        }
    }
    return changes.length;
}

function resetGraphNodeIds(graph) {
    const nodes = getGraphNodes(graph);
    if (!nodes.length) {
        return {
            changed: 0,
            total: 0,
        };
    }

    const orderedNodes = sortNodesForReset(nodes);
    const changed = remapNodeIds(
        graph,
        orderedNodes.map((node, index) => ({ node, nextId: index + 1 })),
    );
    return {
        changed,
        total: orderedNodes.length,
    };
}

function setNodeId(node, nextId) {
    const graph = node?.graph;
    return remapNodeIds(graph, [{ node, nextId }]) > 0;
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
                callback: () => {
                    if (!confirmResetGraphNodeIds(totalNodes)) {
                        return;
                    }

                    try {
                        const result = resetGraphNodeIds(graph);
                        if (!result.changed) {
                            showToast("info", `Current graph already uses IDs 1-${result.total} in the default order.`);
                            return;
                        }
                        showToast("success", `Reassigned ${result.changed} node IDs in ${result.total} nodes. Refresh the page if badges still show old IDs.`);
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
                callback: () => {
                    const rawValue = promptForNodeId(node);
                    if (rawValue === null) {
                        return;
                    }

                    const previousId = node.id;
                    try {
                        const nextId = parseNodeId(rawValue);
                        const updated = setNodeId(node, nextId);
                        if (!updated) {
                            showToast("info", `Node already uses ID ${nextId}.`);
                            return;
                        }
                        showToast("success", `Updated node ID from ${previousId} to ${nextId}. Refresh the page if the badge still shows the old ID.`);
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
