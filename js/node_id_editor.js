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

function moveIndexedEntry(index, previousId, nextId, value) {
    if (!index) {
        return;
    }
    if (index instanceof Map) {
        if (index.get(previousId) === value) {
            index.delete(previousId);
            index.set(nextId, value);
        }
        return;
    }
    if (index[previousId] === value) {
        delete index[previousId];
        index[nextId] = value;
    }
}

function graphHasNodeId(graph, id) {
    return !!graph?.getNodeById?.(id)
        || (graph?._nodes_by_id instanceof Map
            ? graph._nodes_by_id.has(id)
            : Object.prototype.hasOwnProperty.call(graph?._nodes_by_id || {}, id));
}

function getGraphLinks(graph) {
    const links = graph?.links;
    if (!links) {
        return [];
    }
    if (links instanceof Map) {
        return links.values();
    }
    if (Array.isArray(links)) {
        return links;
    }
    return Object.values(links);
}

function getGraphNodes(graph) {
    return Array.isArray(graph?._nodes) ? graph._nodes.filter(Boolean) : [];
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

function compareNodesByPosition(leftNode, rightNode) {
    const left = getNodeLayout(leftNode);
    const right = getNodeLayout(rightNode);

    return left.y - right.y
        || left.x - right.x
        || (leftNode.id ?? 0) - (rightNode.id ?? 0);
}

function sortNodesForReset(nodes) {
    const columnTolerance = getColumnTolerance(nodes);
    const columns = [];

    const nodesByX = [...nodes].sort((leftNode, rightNode) => {
        const left = getNodeLayout(leftNode);
        const right = getNodeLayout(rightNode);
        return left.centerX - right.centerX
            || left.y - right.y
            || (leftNode.id ?? 0) - (rightNode.id ?? 0);
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

    return columns.flatMap((column) => column.nodes.sort(compareNodesByPosition));
}

function updateLinkNodeIds(graph, previousId, nextId) {
    for (const link of getGraphLinks(graph)) {
        if (!link) {
            continue;
        }
        if (Array.isArray(link)) {
            if (link[1] === previousId) {
                link[1] = nextId;
            }
            if (link[3] === previousId) {
                link[3] = nextId;
            }
            continue;
        }

        if (link.origin_id === previousId) {
            link.origin_id = nextId;
        }
        if (link.target_id === previousId) {
            link.target_id = nextId;
        }
        if (link.originId === previousId) {
            link.originId = nextId;
        }
        if (link.targetId === previousId) {
            link.targetId = nextId;
        }
    }
}

function remapLinkNodeIds(graph, idMap) {
    for (const link of getGraphLinks(graph)) {
        if (!link) {
            continue;
        }

        if (Array.isArray(link)) {
            if (idMap.has(link[1])) {
                link[1] = idMap.get(link[1]);
            }
            if (idMap.has(link[3])) {
                link[3] = idMap.get(link[3]);
            }
            continue;
        }

        if (idMap.has(link.origin_id)) {
            link.origin_id = idMap.get(link.origin_id);
        }
        if (idMap.has(link.target_id)) {
            link.target_id = idMap.get(link.target_id);
        }
        if (idMap.has(link.originId)) {
            link.originId = idMap.get(link.originId);
        }
        if (idMap.has(link.targetId)) {
            link.targetId = idMap.get(link.targetId);
        }
    }
}

function updateCanvasSelectionIndexes(graph, previousId, nextId, node) {
    const canvases = new Set([app.canvas, ...(graph?.list_of_graphcanvas || [])].filter(Boolean));
    canvases.forEach((canvas) => {
        moveIndexedEntry(canvas.selected_nodes, previousId, nextId, node);
    });
}

function remapIndexedEntries(index, idMap) {
    if (!index) {
        return;
    }

    if (index instanceof Map) {
        const entries = [];
        for (const [key, value] of index.entries()) {
            entries.push([idMap.get(key) ?? key, value]);
        }
        index.clear();
        for (const [key, value] of entries) {
            index.set(key, value);
        }
        return;
    }

    const nextEntries = {};
    for (const [key, value] of Object.entries(index)) {
        const numericKey = Number(key);
        nextEntries[idMap.get(numericKey) ?? key] = value;
    }

    for (const key of Object.keys(index)) {
        delete index[key];
    }
    Object.assign(index, nextEntries);
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
    const idMap = new Map(orderedNodes.map((node, index) => [node.id, index + 1]));
    const changed = orderedNodes.filter((node, index) => node.id !== index + 1).length;

    if (!changed) {
        return {
            changed: 0,
            total: orderedNodes.length,
        };
    }

    graph.beforeChange?.();
    try {
        for (const node of orderedNodes) {
            node.id = idMap.get(node.id);
        }

        remapIndexedEntries(graph._nodes_by_id, idMap);
        remapLinkNodeIds(graph, idMap);

        const canvases = new Set([app.canvas, ...(graph?.list_of_graphcanvas || [])].filter(Boolean));
        canvases.forEach((canvas) => {
            remapIndexedEntries(canvas.selected_nodes, idMap);
        });

        graph.last_node_id = orderedNodes.length;
    } finally {
        graph.afterChange?.();
    }

    graph.change?.();
    app.canvas?.setDirty?.(true, true);
    return {
        changed,
        total: orderedNodes.length,
    };
}

function setNodeId(node, nextId) {
    const graph = node?.graph;
    const previousId = node?.id;

    if (!graph || !Number.isSafeInteger(previousId)) {
        throw new Error("The selected node is not attached to a graph.");
    }
    if (!Number.isSafeInteger(nextId) || nextId < 1) {
        throw new Error("Node ID must be a positive integer.");
    }
    if (nextId === previousId) {
        return false;
    }
    if (graphHasNodeId(graph, nextId)) {
        throw new Error(`Node ID ${nextId} is already in use.`);
    }

    graph.beforeChange?.();
    try {
        moveIndexedEntry(graph._nodes_by_id, previousId, nextId, node);
        updateLinkNodeIds(graph, previousId, nextId);
        updateCanvasSelectionIndexes(graph, previousId, nextId, node);

        node.id = nextId;
        if (typeof graph.last_node_id === "number") {
            graph.last_node_id = Math.max(graph.last_node_id, nextId);
        }
    } finally {
        graph.afterChange?.();
    }

    graph.change?.();
    node.setDirtyCanvas?.(true, true);
    return true;
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
        if (!node?.graph || !Number.isSafeInteger(node.id)) {
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
