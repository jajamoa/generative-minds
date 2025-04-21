import React, { useCallback, useEffect } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  Panel,
  ConnectionMode,
} from 'reactflow';
import 'reactflow/dist/style.css';

interface NetworkGraphProps {
  nodes: Node[];
  edges: Edge[];
}

const NetworkGraph: React.FC<NetworkGraphProps> = ({ nodes: initialNodes, edges: initialEdges }) => {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  useEffect(() => {
    setNodes(initialNodes);
    setEdges(initialEdges);
  }, [initialNodes, initialEdges, setNodes, setEdges]);

  const onConnect = useCallback((params: any) => {
    setEdges((eds) => [...eds, params]);
  }, [setEdges]);

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        fitView
        fitViewOptions={{ 
          padding: 0.1,
          minZoom: 0.5,
          maxZoom: 1.5,
          duration: 0
        }}
        defaultEdgeOptions={{
          type: 'straight',
          style: { 
            strokeWidth: 1,
            stroke: '#000000',
          }
        }}
        connectionMode={ConnectionMode.Loose}
        nodesDraggable={true}
        nodesConnectable={false}
        elementsSelectable={true}
        minZoom={0.5}
        maxZoom={1.5}
        defaultViewport={{ x: 0, y: 0, zoom: 0.8 }}
        style={{ background: '#ffffff' }}
      >
        <Controls 
          showInteractive={false}
          position="bottom-right"
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '4px',
            padding: '4px',
            backgroundColor: 'white',
            border: '1px solid black',
            borderRadius: '2px',
            bottom: 10,
            right: 10,
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
            borderRadius: '2px',
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