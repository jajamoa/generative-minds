import { Node, Edge } from 'reactflow';
import mermaid from 'mermaid';
import { MarkerType } from 'reactflow';

interface ParsedGraph {
  nodes: Node[];
  edges: Edge[];
}

// Initialize mermaid library with appropriate configuration
mermaid.initialize({
  startOnLoad: false,
  securityLevel: 'loose',
  theme: 'default',
  flowchart: {
    useMaxWidth: false,
    htmlLabels: true
  },
  er: {
    useMaxWidth: false
  }
});

// Helper function to find the central node in a graph
const findCentralNode = (edges: Edge[]): string => {
  const connections = new Map<string, number>();
  edges.forEach(edge => {
    connections.set(edge.source, (connections.get(edge.source) || 0) + 1);
    connections.set(edge.target, (connections.get(edge.target) || 0) + 1);
  });
  return Array.from(connections.entries())
    .sort((a, b) => b[1] - a[1])[0][0];
};

// Calculate node positions in a circular layout
const calculateNodePosition = (nodeId: string, centralNode: string, totalNodes: number, index: number, nodes: Node[]) => {
  if (nodeId === centralNode) {
    return { x: 800, y: 400 }; // Center position
  }

  // Calculate layers based on connection to central node
  const directlyConnected = nodes.length < 15; // Use simpler layout for small graphs
  const angleStep = (2 * Math.PI) / (totalNodes - 1);
  const angle = index * angleStep;
  
  // Radius varies based on graph size
  const baseRadius = Math.min(600, Math.max(300, nodes.length * 20));
  const radius = directlyConnected ? baseRadius : baseRadius * (1 + (index % 2) * 0.5);

  return {
    x: 800 + radius * Math.cos(angle),
    y: 400 + radius * Math.sin(angle)
  };
};

export const parseMermaidFile = async (content: string): Promise<ParsedGraph> => {
  const nodes: Node[] = [];
  const edges: Edge[] = [];
  
  console.log("Processing content:", content);
  
  try {
    // First, use direct text extraction to find all nodes and edges
    const nodeMap = new Map<string, {label: string, className?: string}>();
    const edgeList: {source: string, target: string, label?: string, type?: string}[] = [];
    
    // Clean the content by removing any existing fences and ensuring it's properly formatted
    let cleanContent = content;
    
    // Remove any existing mermaid fences
    cleanContent = cleanContent.replace(/```mermaid\n?/g, '');
    cleanContent = cleanContent.replace(/```\n?/g, '');
    
    // Check if content contains "-->", "--->", "==>", or "--x" which indicates it's likely a graph
    const containsEdges = /--?>|==>|--x/.test(cleanContent);
    
    // Ensure content starts with a valid graph definition
    if (!cleanContent.trim().startsWith('graph') && 
        !cleanContent.trim().startsWith('flowchart')) {
      
      // Detect if it's likely a flowchart by looking for patterns
      if (containsEdges) {
        console.log("Content appears to be a graph but missing declaration, adding flowchart TD");
        cleanContent = 'flowchart TD\n' + cleanContent;
      } else {
        // If not sure, default to flowchart
        console.log("Adding default flowchart TD declaration");
        cleanContent = 'flowchart TD\n' + cleanContent;
      }
    }
    
    console.log("Formatted content for mermaid:", cleanContent);
    
    // Create custom error handler to prevent excessive error outputs
    const originalConsoleError = console.error;
    let mermaidErrorCount = 0;
    const maxMermaidErrors = 1; // Only show one error

    // Temporarily override console.error to limit mermaid errors
    console.error = (...args) => {
      if (args[0] && (String(args[0]).includes('mermaid') || 
                      String(args[0]).includes('Parse error') ||
                      String(args[0]).includes('flowDiagram'))) {
        mermaidErrorCount++;
        if (mermaidErrorCount <= maxMermaidErrors) {
          originalConsoleError("Mermaid parsing error (suppressing additional errors):", args[0]);
        }
        return;
      }
      originalConsoleError(...args); // Pass through non-mermaid errors
    };

    // Try to use mermaid's parse method directly with controlled error output
    try {
      console.log("Attempting to use mermaid.parse directly");
      let parsedData = null;
      try {
        // Wrap in a Promise with timeout to prevent hanging
        const parsePromise = new Promise((resolve, reject) => {
          try {
            const result = mermaid.parse(cleanContent);
            resolve(result);
          } catch (e) {
            reject(e);
          }
        });
        
        // Set timeout to prevent hanging
        parsedData = await Promise.race([
          parsePromise,
          new Promise((_, reject) => setTimeout(() => reject(new Error("Mermaid parse timeout")), 1000))
        ]);
        
        console.log("Mermaid parse completed");
        
        // Try to extract structure if possible
        if (parsedData && typeof parsedData === 'object') {
          console.log("Successfully used mermaid.parse");
          // Processing would go here if we had direct access to the structure
        }
      } catch (parseError) {
        // Silently continue - we'll use regex as fallback
      }
    } catch (e) {
      // Silently continue - we'll use regex as fallback
    }

    // Reset console.error back to original
    console.error = originalConsoleError;
    
    // Continue with existing render approach...
    try {
      console.log("Attempting to parse with mermaid.render...");
      
      // Override console.error again for render attempts
      console.error = (...args) => {
        if (args[0] && (String(args[0]).includes('mermaid') || 
                        String(args[0]).includes('Parse error') ||
                        String(args[0]).includes('flowDiagram'))) {
          mermaidErrorCount++;
          if (mermaidErrorCount <= maxMermaidErrors) {
            originalConsoleError("Mermaid rendering error (suppressing additional errors):", args[0]);
          }
          return;
        }
        originalConsoleError(...args); // Pass through non-mermaid errors
      };
      
      // Wrap render in a Promise with timeout
      const renderPromise = new Promise((resolve, reject) => {
        try {
          mermaid.render('mermaid-graph', cleanContent)
            .then(result => resolve(result))
            .catch(err => reject(err));
        } catch (e) {
          reject(e);
        }
      });
      
      // Set timeout to prevent hanging
      const renderResult = await Promise.race([
        renderPromise,
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error("Mermaid render timeout")), 2000)
        )
      ]);
      
      console.log("Mermaid rendering successful");
      
    } catch (renderError) {
      console.log("Initial mermaid render failed, trying alternative format");
      
      // Try alternative format with minimal error output
      let alternativeSuccess = false;
      
      if (cleanContent.includes('graph TD')) {
        try {
          const altContent = cleanContent.replace('graph TD', 'flowchart TD');
          
          // Try alternative with timeout
          const altRenderPromise = new Promise((resolve, reject) => {
            try {
              mermaid.render('mermaid-graph-alt', altContent)
                .then(result => resolve(result))
                .catch(err => reject(err));
            } catch (e) {
              reject(e);
            }
          });
          
          await Promise.race([
            altRenderPromise,
            new Promise((_, reject) => 
              setTimeout(() => reject(new Error("Alternative render timeout")), 1000)
            )
          ]);
          
          alternativeSuccess = true;
          console.log("Alternative format (flowchart TD) successful");
          
        } catch (altError) {
          // Silently continue
        }
      } else if (cleanContent.includes('flowchart TD')) {
        try {
          const altContent = cleanContent.replace('flowchart TD', 'graph TD');
          
          // Try alternative with timeout
          const altRenderPromise = new Promise((resolve, reject) => {
            try {
              mermaid.render('mermaid-graph-alt', altContent)
                .then(result => resolve(result))
                .catch(err => reject(err));
            } catch (e) {
              reject(e);
            }
          });
          
          await Promise.race([
            altRenderPromise,
            new Promise((_, reject) => 
              setTimeout(() => reject(new Error("Alternative render timeout")), 1000)
            )
          ]);
          
          alternativeSuccess = true;
          console.log("Alternative format (graph TD) successful");
          
        } catch (altError) {
          // Silently continue
        }
      }
      
      if (!alternativeSuccess) {
        console.log("All mermaid parsing attempts failed, falling back to regex parsing");
      }
    } finally {
      // Ensure we reset console.error
      console.error = originalConsoleError;
    }
    
    // Now we need to extract node and edge information from another approach 
    // since mermaid's internal structure isn't directly accessible
    
    // Extract all node definitions - A1["Label"] format
    const nodeDefinitionRegex = /([A-Za-z0-9_-]+)\s*\[(\"?)([^\]]*?)(\2)\]/g;
    let match;
    let contentForMatching = content;
    
    while ((match = nodeDefinitionRegex.exec(contentForMatching)) !== null) {
      const [, id, , label] = match;
      nodeMap.set(id, { label });
      console.log(`Found node definition: ${id} with label "${label}"`);
    }
    
    // Extract all connections - A1 --> A2 format
    const edgeRegex = /([A-Za-z0-9_-]+)\s*(-->|==>|--x)\s*(?:\|(.*?)\|)?\s*([A-Za-z0-9_-]+)/g;
    contentForMatching = content;
    
    while ((match = edgeRegex.exec(contentForMatching)) !== null) {
      const [, source, type, label, target] = match;
      
      // Add nodes if they don't exist
      if (!nodeMap.has(source)) {
        nodeMap.set(source, { label: source });
        console.log(`Implied node from edge: ${source}`);
      }
      
      if (!nodeMap.has(target)) {
        nodeMap.set(target, { label: target });
        console.log(`Implied node from edge: ${target}`);
      }
      
      edgeList.push({ source, target, label, type });
      console.log(`Found edge: ${source} ${type} ${target}`);
    }
    
    // Process lines that might contain both node definitions and edges
    const complexLineRegex = /([A-Za-z0-9_-]+)(?:\s*\[(\"?)([^\]]*?)(\2)\])?\s*(-->|==>|--x)\s*(?:\|(.*?)\|)?\s*([A-Za-z0-9_-]+)(?:\s*\[(\"?)([^\]]*?)(\8)\])?/g;
    contentForMatching = content;
    
    while ((match = complexLineRegex.exec(contentForMatching)) !== null) {
      const [, sourceId, , sourceLabel, , edgeType, edgeLabel, targetId, , targetLabel] = match;
      
      // Add source node if it has a label and isn't already in the map
      if (sourceLabel && (!nodeMap.has(sourceId) || !nodeMap.get(sourceId)?.label)) {
        nodeMap.set(sourceId, { label: sourceLabel });
        console.log(`Found combined source node: ${sourceId} with label "${sourceLabel}"`);
      }
      
      // Add target node if it has a label and isn't already in the map
      if (targetLabel && (!nodeMap.has(targetId) || !nodeMap.get(targetId)?.label)) {
        nodeMap.set(targetId, { label: targetLabel });
        console.log(`Found combined target node: ${targetId} with label "${targetLabel}"`);
      }
      
      // Ensure both nodes exist even without labels
      if (!nodeMap.has(sourceId)) {
        nodeMap.set(sourceId, { label: sourceId });
      }
      
      if (!nodeMap.has(targetId)) {
        nodeMap.set(targetId, { label: targetId });
      }
      
      // Add edge if not already in the list
      if (!edgeList.some(e => e.source === sourceId && e.target === targetId)) {
        edgeList.push({ 
          source: sourceId, 
          target: targetId, 
          label: edgeLabel, 
          type: edgeType 
        });
        console.log(`Found complex edge: ${sourceId} ${edgeType} ${targetId}`);
      }
    }
    
    // If we don't have enough nodes or edges, try a more relaxed regex approach
    if (nodeMap.size === 0 || edgeList.length === 0) {
      console.log("Initial parsing didn't find enough elements, trying more relaxed regex");
      
      // Fallback to pure regex approach to extract nodes and edges
      // (This is our existing manual parsing logic moved here as fallback)
      
      // Extract node IDs from the content - support both [label] and ["label"] formats
      const nodeIdRegex = /([A-Za-z0-9_-]+)\s*\[([^\]]*)\]/g;
      const nodeLabels = new Map<string, string>();
      
      let contentCopy = content;
      let fallbackMatch;
      while ((fallbackMatch = nodeIdRegex.exec(contentCopy)) !== null) {
        const id = fallbackMatch[1];
        const label = fallbackMatch[2].replace(/"/g, '');
        nodeMap.set(id, { label });
        console.log(`Extracted node: ${id} with label: ${label}`);
      }
      
      // Extract relationships from the content - more relaxed pattern
      const edgeRegexGlobal = /([A-Za-z0-9_-]+)\s*(-->|==>|--x|-.->)\s*([A-Za-z0-9_-]+)/g;
      contentCopy = content;
      while ((fallbackMatch = edgeRegexGlobal.exec(contentCopy)) !== null) {
        const source = fallbackMatch[1];
        const target = fallbackMatch[3];
        const type = fallbackMatch[2];
        
        // Ensure nodes exist even if they don't have a label definition
        if (!nodeMap.has(source)) {
          nodeMap.set(source, { label: source });
        }
        if (!nodeMap.has(target)) {
          nodeMap.set(target, { label: target });
        }
        
        edgeList.push({ source, target, type });
        console.log(`Extracted edge: ${source} -> ${target}`);
      }
    }
    
    // If still no success, try an even simpler approach for basic node-to-node connections
    if (nodeMap.size === 0 || edgeList.length === 0) {
      console.log("Trying ultra-simplified parser for basic connections");
      
      // Look for any alphanumeric sequence followed by -->
      const simpleNodeRegex = /([A-Za-z0-9_-]+)\s*-->/g;
      let contentCopy = content;
      let simpleMatch;
      
      // First pass - collect all node IDs
      while ((simpleMatch = simpleNodeRegex.exec(contentCopy)) !== null) {
        const id = simpleMatch[1];
        if (!nodeMap.has(id)) {
          nodeMap.set(id, { label: id });
          console.log(`Found simple node: ${id}`);
        }
      }
      
      // Second pass - find target nodes (after -->)
      const simpleEdgeRegex = /([A-Za-z0-9_-]+)\s*-->\s*([A-Za-z0-9_-]+)/g;
      contentCopy = content;
      
      while ((simpleMatch = simpleEdgeRegex.exec(contentCopy)) !== null) {
        const source = simpleMatch[1];
        const target = simpleMatch[2];
        
        // Add nodes if needed
        if (!nodeMap.has(source)) {
          nodeMap.set(source, { label: source });
        }
        if (!nodeMap.has(target)) {
          nodeMap.set(target, { label: target });
        }
        
        // Add edge if it doesn't exist
        if (!edgeList.some(e => e.source === source && e.target === target)) {
          edgeList.push({ source, target, type: '-->' });
          console.log(`Found simple edge: ${source} --> ${target}`);
        }
      }
    }
    
    // Last resort: line-by-line parsing for completely custom formats
    if (nodeMap.size === 0 || edgeList.length === 0) {
      console.log("Trying line-by-line parsing as last resort");
      
      // Split content by lines and process each line individually
      const lines = content.split('\n').map(line => line.trim()).filter(line => line);
      
      for (const line of lines) {
        // Skip graph definition lines
        if (line.startsWith('graph') || line.startsWith('flowchart')) {
          continue;
        }
        
        // Try to identify if line has a connection symbol
        if (line.includes('-->') || line.includes('==>') || line.includes('--x')) {
          // Extract potential node IDs by splitting around connection symbols
          const parts = line.split(/\s*(-->|==>|--x)\s*/);
          
          // We should have at least 3 parts (source, connection type, target)
          if (parts.length >= 3) {
            for (let i = 0; i < parts.length - 2; i += 2) {
              const source = parts[i].trim();
              const connectionType = parts[i + 1];
              const target = parts[i + 2].trim();
              
              // Extract source ID (removing any label part)
              let sourceId = source;
              let sourceLabel = sourceId;
              
              const sourceMatch = source.match(/^([A-Za-z0-9_-]+)(?:\s*\[(.*?)\])?/);
              if (sourceMatch) {
                sourceId = sourceMatch[1];
                if (sourceMatch[2]) {
                  sourceLabel = sourceMatch[2].replace(/"/g, '');
                }
              }
              
              // Extract target ID (removing any label part)
              let targetId = target;
              let targetLabel = targetId;
              
              const targetMatch = target.match(/^([A-Za-z0-9_-]+)(?:\s*\[(.*?)\])?/);
              if (targetMatch) {
                targetId = targetMatch[1];
                if (targetMatch[2]) {
                  targetLabel = targetMatch[2].replace(/"/g, '');
                }
              }
              
              // Add nodes if valid IDs and not already in the map
              if (sourceId && !sourceId.includes(' ') && !sourceId.includes('-->')) {
                nodeMap.set(sourceId, { label: sourceLabel });
              }
              
              if (targetId && !targetId.includes(' ') && !targetId.includes('-->')) {
                nodeMap.set(targetId, { label: targetLabel });
              }
              
              // Add edge if valid
              if (sourceId && targetId && 
                  !sourceId.includes(' ') && !targetId.includes(' ') &&
                  !edgeList.some(e => e.source === sourceId && e.target === targetId)) {
                edgeList.push({
                  source: sourceId,
                  target: targetId,
                  type: connectionType
                });
                console.log(`Found line-parsed edge: ${sourceId} ${connectionType} ${targetId}`);
              }
            }
          }
        }
      }
    }
    
    // Create React Flow nodes from the collected node data
    nodeMap.forEach((data, id) => {
      nodes.push({
        id,
        data: {
          label: data.label,
          className: data.className,
          width: data.label.length * 8,
        },
        position: { x: 0, y: 0 }, // Will position later
        type: 'default',
        style: {
          width: 'auto',
          padding: '12px 16px',
          fontSize: '13px',
          border: '1px solid #000',
          borderRadius: '2px',
          background: '#fff',
          boxShadow: '2px 2px 0 rgba(0,0,0,0.1)',
        }
      });
    });
    
    // Create React Flow edges
    edgeList.forEach(({ source, target, label, type }) => {
      const isNegative = type === '--x';
      edges.push({
        id: `e${source}-${target}`,
        source,
        target,
        label,
        type: 'default',
        animated: false,
        markerEnd: {
          type: MarkerType.ArrowClosed,
          width: 13,
          height: 13,
          color: isNegative ? '#ff0000' : '#2E8B57',
        },
        style: {
          stroke: isNegative ? '#ff0000' : '#2E8B57',
          strokeWidth: 1,
          strokeDasharray: isNegative ? '5,5' : undefined,
        }
      });
    });

    // If we have nodes and edges, position them
    if (nodes.length > 0 && edges.length > 0) {
      // Find central node and position all nodes
      const centralNode = findCentralNode(edges);
      nodes.forEach((node, index) => {
        if (node.id !== centralNode) {
          const position = calculateNodePosition(
            node.id,
            centralNode,
            nodes.length,
            index,
            nodes
          );
          node.position = position;
        } else {
          node.position = { x: 800, y: 400 };
        }
      });
    }
    
    console.log("Final parse results:");
    console.log("Nodes:", nodes.length, nodes);
    console.log("Edges:", edges.length, edges);

    return { nodes, edges };
    
  } catch (error) {
    console.error("Error parsing mermaid file:", error);
    // Return empty result on error
    return { nodes: [], edges: [] };
  }
};

export const readFileAsText = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target?.result as string);
    reader.onerror = (e) => reject(e);
    reader.readAsText(file);
  });
}; 
