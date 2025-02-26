import { useEffect, useRef } from 'react';
import PropTypes from 'prop-types';
import { Network } from 'vis-network';
import { DataSet } from 'vis-data';

const CausalGraph = ({ nodes, edges }) => {
  const containerRef = useRef(null);
  const networkRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // 创建数据集
    const nodesDataSet = new DataSet(nodes);
    const edgesDataSet = new DataSet(edges);

    // 配置选项
    const options = {
      nodes: {
        shape: 'box',
        margin: 10,
        font: {
          size: 14,
          color: '#ffffff',
          face: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
          strokeWidth: 0
        },
        borderWidth: 1,
        color: {
          background: 'rgba(255, 255, 255, 0.15)',
          border: 'rgba(255, 255, 255, 0.35)',
          highlight: {
            background: 'rgba(255, 255, 255, 0.25)',
            border: 'rgba(255, 255, 255, 0.45)'
          }
        }
      },
      edges: {
        arrows: 'to',
        color: {
          color: 'rgba(255, 255, 255, 0.35)',
          highlight: 'rgba(255, 255, 255, 0.45)',
          hover: 'rgba(255, 255, 255, 0.45)'
        },
        width: 1,
        smooth: {
          type: 'continuous'
        }
      },
      physics: {
        enabled: true,
        hierarchicalRepulsion: {
          nodeDistance: 250,
          springLength: 300,
          springConstant: 0.01,
          damping: 0.09
        },
        solver: 'hierarchicalRepulsion',
        stabilization: {
          enabled: true,
          iterations: 1000,
          updateInterval: 100
        }
      },
      layout: {
        hierarchical: {
          enabled: true,
          direction: 'LR',
          sortMethod: 'directed',
          levelSeparation: 300,
          nodeSpacing: 200,
          treeSpacing: 200
        }
      },
      interaction: {
        hover: true,
        tooltipDelay: 200
      }
    };

    // 创建网络
    networkRef.current = new Network(
      containerRef.current,
      { nodes: nodesDataSet, edges: edgesDataSet },
      options
    );

    return () => {
      if (networkRef.current) {
        networkRef.current.destroy();
        networkRef.current = null;
      }
    };
  }, [nodes, edges]);

  return (
    <div 
      ref={containerRef} 
      style={{ 
        height: '800px', 
        width: '100%',
        border: '1px solid rgba(255, 255, 255, 0.25)',
        borderRadius: 0,
        background: 'rgba(0, 0, 0, 0.85)'
      }} 
    />
  );
};

CausalGraph.propTypes = {
  nodes: PropTypes.arrayOf(PropTypes.shape({
    id: PropTypes.string.isRequired,
    label: PropTypes.string.isRequired
  })).isRequired,
  edges: PropTypes.arrayOf(PropTypes.shape({
    from: PropTypes.string.isRequired,
    to: PropTypes.string.isRequired
  })).isRequired
};

export default CausalGraph; 