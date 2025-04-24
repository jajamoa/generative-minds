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
  // Support for graph TD format and regular format
  const graphType = lines.find(line => /^\s*graph (TD|LR|RL|BT);?/.test(line));
  
  // Support two node formats: n0["label"] and n0["label"]:::classname
  // Use a regex that allows for multi-line labels with newlines
  const nodePattern = /\s*(n\d+)\[\"?([\s\S]*?)\"?\](?:::(\w+))?/;
  const edgePattern = /\s*(n\d+)\s*(==>|-->|--x)\s*(?:\|(.*?)\|)?\s*(n\d+)/;
  const stylePattern = /\s*linkStyle\s+(\d+)\s+stroke:(#[A-Fa-f0-9]{6}),.*stroke-width:(\d+)px/;
  const classDefPattern = /\s*classDef\s+(\w+)\s+(.*?);?$/;

  // Process class definitions
  const classStyles: Record<string, any> = {};
  lines.forEach(line => {
    const classMatch = line.match(classDefPattern);
    if (classMatch) {
      const [, className, styleText] = classMatch;
      const styleObj: Record<string, string> = {};
      
      // Parse CSS style attributes
      const styleParts = styleText.split(',');
      styleParts.forEach(part => {
        const [property, value] = part.split(':').map(s => s.trim());
        if (property && value) {
          styleObj[property] = value;
        }
      });
      
      classStyles[className] = styleObj;
    }
  });

  // Process node definitions in the content, accounting for multi-line labels
  let i = 0;
  while (i < lines.length) {
    let currentLine = lines[i];
    let nodeMatch = currentLine.match(/\s*(n\d+)\[\"?(.*?)(?:\"?\](?:::(\w+))?)?$/);
    
    // If line starts a node definition but doesn't end it
    if (nodeMatch && !currentLine.includes(']')) {
      let fullNodeText = currentLine;
      let j = i + 1;
      
      // Continue reading lines until we find the closing bracket
      while (j < lines.length && !lines[j].includes(']')) {
        fullNodeText += '\n' + lines[j];
        j++;
      }
      
      // Add the line with the closing bracket
      if (j < lines.length) {
        fullNodeText += '\n' + lines[j];
        i = j; // Update the outer loop index
      }
      
      // Now try to match the complete multi-line node definition
      const completeMatch = fullNodeText.match(/\s*(n\d+)\[\"?([\s\S]*?)\"?\](?:::(\w+))?/);
      if (completeMatch) {
        const [, id, rawLabel, className] = completeMatch;
        
        // Base style
        const baseStyle = {
          width: 'auto',
          padding: '12px 16px',
          fontSize: '13px',
          border: '1px solid #000',
          borderRadius: '2px',
          background: '#fff',
          boxShadow: '2px 2px 0 rgba(0,0,0,0.1)',
        };
        
        // Merge class styles
        let nodeStyle = { ...baseStyle };
        if (className && classStyles[className]) {
          // Convert Mermaid style properties to React style properties
          const cssStyle = classStyles[className];
          if (cssStyle.fill) nodeStyle.background = cssStyle.fill;
          if (cssStyle.stroke) nodeStyle.border = `1px solid ${cssStyle.stroke}`;
          if (cssStyle['stroke-width']) {
            const width = parseInt(cssStyle['stroke-width']);
            if (!isNaN(width)) nodeStyle.border = `${width}px solid ${cssStyle.stroke || '#000'}`;
          }
        }
        
        // Clean up label by removing whitespace at the beginning of each line
        const label = rawLabel.split('\n')
          .map(line => line.trim())
          .join('\n')
          .replace(/[\[\]]/g, '');
        
        nodes.push({
          id: id,
          data: { 
            label: label,
            className: className, // Store class name for future use
            width: Math.max(...label.split('\n').map(l => l.length)) * 8, // Approximate width based on longest line
          },
          position: { x: 0, y: 0 }, // Temporary position
          type: 'default',
          style: nodeStyle
        });
      }
    } else {
      // Handle single-line node definitions
      if (nodeMatch && currentLine.includes(']')) {
        const [, id, label, className] = nodeMatch;
        
        // Base style
        const baseStyle = {
          width: 'auto',
          padding: '12px 16px',
          fontSize: '13px',
          border: '1px solid #000',
          borderRadius: '2px',
          background: '#fff',
          boxShadow: '2px 2px 0 rgba(0,0,0,0.1)',
        };
        
        // Merge class styles
        let nodeStyle = { ...baseStyle };
        if (className && classStyles[className]) {
          // Convert Mermaid style properties to React style properties
          const cssStyle = classStyles[className];
          if (cssStyle.fill) nodeStyle.background = cssStyle.fill;
          if (cssStyle.stroke) nodeStyle.border = `1px solid ${cssStyle.stroke}`;
          if (cssStyle['stroke-width']) {
            const width = parseInt(cssStyle['stroke-width']);
            if (!isNaN(width)) nodeStyle.border = `${width}px solid ${cssStyle.stroke || '#000'}`;
          }
        }
        
        nodes.push({
          id: id,
          data: { 
            label: label.replace(/[\[\]]/g, ''),
            className: className, // Store class name for future use
            width: label.length * 8, // Approximate width based on text length
          },
          position: { x: 0, y: 0 }, // Temporary position
          type: 'default',
          style: nodeStyle
        });
      }
      i++;
    }
  }

  // Collect edges and their styles
  lines.forEach((line, index) => {
    const edgeMatch = line.match(edgePattern);
    if (edgeMatch) {
      const [, source, type, label, target] = edgeMatch;
      edges.push({
        id: `e${source}-${target}`,
        source: source,
        target: target,
        label: label,
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