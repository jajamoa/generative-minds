import { Node, Edge } from 'reactflow';

interface ParsedGraph {
  nodes: Node[];
  edges: Edge[];
}

const findCentralNode = (edges: Edge[]): string => {
  const connections = new Map<string, number>();
  edges.forEach(edge => {
    connections.set(edge.source, (connections.get(edge.source) || 0) + 1);
    connections.set(edge.target, (connections.get(edge.target) || 0) + 1);
  });
  return Array.from(connections.entries())
    .sort((a, b) => b[1] - a[1])[0][0];
};

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

export const parseMermaidFile = (content: string): ParsedGraph => {
  const nodes: Node[] = [];
  const edges: Edge[] = [];
  
  const lines = content.split('\n');
  const nodePattern = /\s*n(\d+)\[(.*?)\]/;
  const edgePattern = /\s*n(\d+)\s*(-->|--x)\s*n(\d+)/;
  const stylePattern = /\s*linkStyle\s+(\d+)\s+stroke:(#[A-Fa-f0-9]{6}),.*stroke-width:(\d+)px/;

  // First pass: collect all nodes
  const nodeLines = lines.filter(line => nodePattern.test(line));
  
  // Collect nodes first
  nodeLines.forEach((line) => {
    const nodeMatch = line.match(nodePattern);
    if (nodeMatch) {
      const [, id, label] = nodeMatch;
      nodes.push({
        id: `n${id}`,
        data: { 
          label: label.replace(/[\[\]]/g, ''),
          width: label.length * 8, // Approximate width based on text length
        },
        position: { x: 0, y: 0 }, // Temporary position
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
    }
  });

  // Collect edges and their styles
  lines.forEach((line, index) => {
    const edgeMatch = line.match(edgePattern);
    if (edgeMatch) {
      const [, source, type, target] = edgeMatch;
      edges.push({
        id: `e${source}-${target}`,
        source: `n${source}`,
        target: `n${target}`,
        type: 'default',
        animated: false,
        style: {
          stroke: type === '--x' ? '#ff0000' : '#000000',
          strokeWidth: 1,
          strokeDasharray: type === '--x' ? '5,5' : undefined,
        }
      });
    }
  });

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

  return { nodes, edges };
};

export const readFileAsText = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target?.result as string);
    reader.onerror = (e) => reject(e);
    reader.readAsText(file);
  });
}; 