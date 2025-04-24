import React, { useCallback, useEffect, useMemo } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  Panel,
  ConnectionMode,
  MarkerType,
  getBezierPath,
  EdgeProps,
} from 'reactflow';
import 'reactflow/dist/style.css';

// Add type definition with support for className
interface CustomNodeData {
  label: string;
  width?: number;
  className?: string;
}

// Define common style mappings
const CLASS_STYLES: Record<string, React.CSSProperties> = {
  stanceNode: {
    background: '#f9f', // Pink fill
    border: '2px solid #333',
    minWidth: '180px',
  },
  factorNode: {
    background: '#bbf', // Light blue fill
    border: '1px solid #333',
    minWidth: '200px',
    maxWidth: '400px',
    whiteSpace: 'pre-wrap' as const,
  },
  title: {
    background: 'none',
    border: 'none',
    fontWeight: 'bold',
    fontSize: '16px',
    padding: '5px 10px',
  }
};

interface NetworkGraphProps {
  nodes: Node[];
  edges: Edge[];
  layout?: 'default' | 'force' | 'tree';
}

// Improved tree layout algorithm
const applyTreeLayout = (nodes: Node[], edges: Edge[]): Node[] => {
  if (nodes.length === 0) return [];
  
  // Create deep copies of nodes to avoid mutating the original
  const nodesCopy = nodes.map(node => ({
    ...node,
    position: { ...node.position },
  }));

  // Build node lookup map
  const nodeMap = new Map<string, Node>();
  nodesCopy.forEach(node => nodeMap.set(node.id, node));
  
  // Build adjacency list for the graph
  const adjacencyList = new Map<string, Set<string>>();
  edges.forEach(edge => {
    if (!adjacencyList.has(edge.source)) {
      adjacencyList.set(edge.source, new Set());
    }
    if (!adjacencyList.has(edge.target)) {
      adjacencyList.set(edge.target, new Set());
    }
    
    // Add connections (source to target only for tree direction)
    adjacencyList.get(edge.source)?.add(edge.target);
  });
  
  // Find root node (node without incoming edges)
  let rootId = '';
  
  // Find all nodes with incoming edges
  const incomingEdges = new Map<string, number>();
  edges.forEach(edge => {
    incomingEdges.set(edge.target, (incomingEdges.get(edge.target) || 0) + 1);
  });
  
  // Nodes without incoming edges are potential roots
  const potentialRoots = nodesCopy.filter(node => !incomingEdges.has(node.id));
  
  if (potentialRoots.length > 0) {
    rootId = potentialRoots[0].id;
  } else if (nodesCopy.length > 0) {
    rootId = nodesCopy[0].id;
  }
  
  // Use BFS to assign node levels
  const visited = new Set<string>();
  const levels = new Map<string, number>();
  const levelNodes = new Map<number, string[]>();
  
  const queue = [rootId];
  levels.set(rootId, 0);
  if (!levelNodes.has(0)) levelNodes.set(0, []);
  levelNodes.get(0)?.push(rootId);
  visited.add(rootId);
  
  while (queue.length > 0) {
    const currentId = queue.shift()!;
    const currentLevel = levels.get(currentId)!;
    const nextLevel = currentLevel + 1;
    
    const neighbors = adjacencyList.get(currentId);
    if (neighbors) {
      // Convert Set to Array to avoid iteration issues
      Array.from(neighbors).forEach(neighborId => {
        if (!visited.has(neighborId)) {
          visited.add(neighborId);
          levels.set(neighborId, nextLevel);
          
          if (!levelNodes.has(nextLevel)) {
            levelNodes.set(nextLevel, []);
          }
          levelNodes.get(nextLevel)?.push(neighborId);
          
          queue.push(neighborId);
        }
      });
    }
  }
  
  // Assign levels to any disconnected nodes
  nodesCopy.forEach(node => {
    if (!visited.has(node.id)) {
      // Find max level and add unconnected nodes to the next level
      const levelsArray = Array.from(levels.values());
      const maxLevel = levelsArray.length > 0 ? Math.max(...levelsArray) : 0;
      const specialLevel = maxLevel + 1;
      
      levels.set(node.id, specialLevel);
      
      if (!levelNodes.has(specialLevel)) {
        levelNodes.set(specialLevel, []);
      }
      levelNodes.get(specialLevel)?.push(node.id);
    }
  });
  
  // Calculate positions for each node by level
  const levelCount = levelNodes.size;
  const verticalSpacing = 120; // Reduced vertical spacing
  const centerX = 400;
  const startY = 80;
  
  // Place nodes by level (top to bottom)
  for (let level = 0; level < levelCount; level++) {
    const nodesInLevel = levelNodes.get(level) || [];
    const nodeHorizontalSpacing = 200; // Reduced horizontal spacing
    
    // Calculate total width needed for this level
    const levelWidth = nodesInLevel.length * nodeHorizontalSpacing;
    const startX = centerX - levelWidth / 2 + nodeHorizontalSpacing / 2;
    
    // Position each node in this level
    nodesInLevel.forEach((nodeId, index) => {
      const node = nodeMap.get(nodeId);
      if (node) {
        node.position = {
          x: startX + index * nodeHorizontalSpacing,
          y: startY + level * verticalSpacing
        };
      }
    });
  }
  
  return Array.from(nodeMap.values());
};

// Improved force-directed layout algorithm
const applyForceLayout = (nodes: Node[], edges: Edge[]): Node[] => {
  if (nodes.length === 0) return [];
  
  // Create a deep copy of nodes to avoid mutating the original
  const nodesCopy = nodes.map(node => ({
    ...node,
    position: { ...node.position },
  }));
  
  // Center point for the layout
  const centerX = 800;
  const centerY = 400;
  
  // Create a map for quick node lookup
  const nodeMap = new Map<string, Node>();
  nodesCopy.forEach(node => nodeMap.set(node.id, node));
  
  // Initialize node positions in a circle if needed
  const radius = Math.min(600, Math.max(300, nodes.length * 20));
  nodesCopy.forEach((node, index) => {
    const angle = (index / nodesCopy.length) * 2 * Math.PI;
    node.position = {
      x: centerX + radius * Math.cos(angle),
      y: centerY + radius * Math.sin(angle)
    };
  });
  
  // Create a map of connected nodes
  const connections = new Map<string, string[]>();
  edges.forEach(edge => {
    if (!connections.has(edge.source)) {
      connections.set(edge.source, []);
    }
    connections.get(edge.source)?.push(edge.target);
    
    if (!connections.has(edge.target)) {
      connections.set(edge.target, []);
    }
    connections.get(edge.target)?.push(edge.source);
  });
  
  // Constants for force simulation
  const iterations = 100;
  const k = Math.sqrt(1000000 / nodes.length); // Optimal distance
  const gravity = 0.05;
  const initialDamping = 0.8;
  
  // Force-directed algorithm (Fruchterman-Reingold)
  for (let i = 0; i < iterations; i++) {
    // Calculate cooling factor (decreases with iterations)
    const damping = initialDamping * (1 - i / iterations);
    
    // Calculate repulsive forces
    const displacement = new Map<string, { dx: number, dy: number }>();
    nodesCopy.forEach(node => {
      displacement.set(node.id, { dx: 0, dy: 0 });
    });
    
    // Apply repulsive forces between all pairs of nodes
    for (let i = 0; i < nodesCopy.length; i++) {
      const node1 = nodesCopy[i];
      const disp1 = displacement.get(node1.id)!;
      
      for (let j = i + 1; j < nodesCopy.length; j++) {
        const node2 = nodesCopy[j];
        const disp2 = displacement.get(node2.id)!;
        
        // Calculate distance and direction
        const dx = node1.position.x - node2.position.x;
        const dy = node1.position.y - node2.position.y;
        const distance = Math.max(0.1, Math.sqrt(dx * dx + dy * dy));
        
        // Calculate repulsive force (inversely proportional to distance)
        const force = k * k / distance;
        
        // Calculate force components
        const fx = force * dx / distance;
        const fy = force * dy / distance;
        
        // Apply to both nodes in opposite directions
        disp1.dx += fx;
        disp1.dy += fy;
        disp2.dx -= fx;
        disp2.dy -= fy;
      }
    }
    
    // Apply attractive forces between connected nodes
    edges.forEach(edge => {
      const source = nodeMap.get(edge.source);
      const target = nodeMap.get(edge.target);
      
      if (source && target) {
        const dispSource = displacement.get(edge.source)!;
        const dispTarget = displacement.get(edge.target)!;
        
        // Calculate distance and direction
        const dx = source.position.x - target.position.x;
        const dy = source.position.y - target.position.y;
        const distance = Math.max(0.1, Math.sqrt(dx * dx + dy * dy));
        
        // Calculate attractive force (proportional to distance)
        const force = distance * distance / k;
        
        // Calculate force components
        const fx = force * dx / distance;
        const fy = force * dy / distance;
        
        // Apply to both nodes (attraction pulls them together)
        dispSource.dx -= fx;
        dispSource.dy -= fy;
        dispTarget.dx += fx;
        dispTarget.dy += fy;
      }
    });
    
    // Apply gravitational force toward center and update positions
    nodesCopy.forEach(node => {
      const disp = displacement.get(node.id)!;
      
      // Add gravity toward center
      disp.dx -= (node.position.x - centerX) * gravity;
      disp.dy -= (node.position.y - centerY) * gravity;
      
      // Calculate magnitude of displacement
      const magnitude = Math.sqrt(disp.dx * disp.dx + disp.dy * disp.dy);
      
      // Limit maximum displacement using damping
      const limitedMagnitude = Math.min(magnitude, 15 * damping);
      
      // Apply displacement
      if (magnitude > 0) {
        node.position.x += disp.dx / magnitude * limitedMagnitude;
        node.position.y += disp.dy / magnitude * limitedMagnitude;
      }
    });
  }
  
  return nodesCopy;
};

// Custom node component for handling multiline text
const MultiLineNode = ({ data }: { data: any }) => {
  const lines = data.label.split('\n');
  
  return (
    <div style={{ padding: '10px', textAlign: 'center' }}>
      {lines.map((line: string, i: number) => (
        <div key={i}>{line}</div>
      ))}
    </div>
  );
};

const NetworkGraph: React.FC<NetworkGraphProps> = ({ 
  nodes: initialNodes, 
  edges: initialEdges,
  layout = 'default' 
}) => {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Apply different layouts based on the selected option
  useEffect(() => {
    let positionedNodes: Node[];
    
    try {
      switch (layout) {
        case 'force':
          positionedNodes = applyForceLayout(initialNodes, initialEdges);
          break;
        case 'tree':
          positionedNodes = applyTreeLayout(initialNodes, initialEdges);
          break;
        default:
          // Use the original positioning from the parser
          positionedNodes = initialNodes;
      }
    } catch (error) {
      console.error("Error applying layout:", error);
      positionedNodes = initialNodes;  // Fallback to default on error
    }
    
    // Apply class-based styling to nodes and handle multi-line labels
    const styledNodes = positionedNodes.map(node => {
      // Get className from node data
      const nodeData = node.data as CustomNodeData;
      const className = nodeData.className;
      let nodeStyle = { ...node.style };
      
      // Check if the label contains newlines
      const hasMultipleLines = nodeData.label.includes('\n');
      
      // If className exists and there's a matching style definition, apply it
      if (className && CLASS_STYLES[className]) {
        nodeStyle = {
          ...nodeStyle,
          ...CLASS_STYLES[className]
        };
      }
      
      // Add styles for multi-line text
      if (hasMultipleLines) {
        nodeStyle = {
          ...nodeStyle,
          whiteSpace: 'pre-wrap',
          textAlign: 'center',
          width: 'auto',
          minWidth: '180px',
          maxWidth: '400px',
        };
      }
      
      return {
        ...node,
        style: nodeStyle
      };
    });
    
    // Preserve edge styles
    const styledEdges = initialEdges;
    
    setNodes(styledNodes);
    setEdges(styledEdges);
  }, [initialNodes, initialEdges, layout, setNodes, setEdges]);

  const onConnect = useCallback((params: any) => {
    setEdges((eds) => [...eds, params]);
  }, [setEdges]);

  // Base node style
  const baseNodeStyle = useMemo(() => {
    return {
      width: 'auto',
      padding: '12px 16px',
      fontSize: '13px',
      border: '1px solid #000',
      borderRadius: '4px',
      background: '#fff',
      boxShadow: '2px 2px 0 rgba(0,0,0,0.1)',
      textAlign: 'center' as const,
      whiteSpace: 'pre-wrap' as const,
      overflow: 'hidden',
    };
  }, []);

  // Register custom node types
  const nodeTypes = useMemo(() => ({
    multiline: MultiLineNode,
  }), []);

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ 
          padding: 0.2,
          minZoom: 0.5,
          maxZoom: 1.5,
          duration: 0
        }}
        defaultEdgeOptions={{
          type: 'straight',
          style: { 
            strokeWidth: 1,
          }
        }}
        connectionMode={ConnectionMode.Loose}
        nodesDraggable={true}
        nodesConnectable={false}
        elementsSelectable={true}
        minZoom={0.25}
        maxZoom={2.0}
        defaultViewport={{ x: 0, y: 0, zoom: 0.8 }}
        style={{ background: '#ffffff' }}
      >
        <Controls 
          showInteractive={false}
          position="bottom-left"
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '4px',
            padding: '4px',
            backgroundColor: 'white',
            border: '1px solid black',
            borderRadius: '4px',
            bottom: 10,
            left: 10,
          }}
        />
        <Background 
          color="#f0f0f0" 
          gap={16} 
          size={1}
          style={{
            backgroundColor: '#ffffff',
          }}
        />
        <Panel 
          position="top-left" 
          style={{ 
            fontSize: 12, 
            color: '#666',
            padding: '6px 8px',
            backgroundColor: 'white',
            border: '1px solid #000',
            borderRadius: '4px',
            top: 10,
            left: 10,
          }}
        >
          Drag to move nodes | Scroll to zoom | Hold right click to pan
        </Panel>
      </ReactFlow>
    </div>
  );
};

export default NetworkGraph; 