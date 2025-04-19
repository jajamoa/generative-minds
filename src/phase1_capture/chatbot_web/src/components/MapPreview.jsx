import { useEffect, useRef } from 'react'
import PropTypes from 'prop-types'
import mapboxgl from 'mapbox-gl'
import '@mapbox/mapbox-gl-geocoder/dist/mapbox-gl-geocoder.css'
import MapboxGeocoder from '@mapbox/mapbox-gl-geocoder'
import 'mapbox-gl/dist/mapbox-gl.css'

// Use environment variable for token
mapboxgl.accessToken = import.meta.env.VITE_MAPBOX_TOKEN

function MapPreview({ onLocationSelect }) {
  const mapContainer = useRef(null)
  const map = useRef(null)
  const marker = useRef(null)

  const flyToLocation = (coordinates, zoom = 16) => {
    if (map.current) {
      map.current.flyTo({
        center: coordinates,
        zoom: zoom,
        essential: true,
        duration: 1500, // 增加动画时间
        padding: 20 // 添加一些内边距
      })
    }
  }

  useEffect(() => {
    if (!mapboxgl.accessToken) {
      console.error('Mapbox token is not set')
      return
    }
    console.log('Mapbox token:', mapboxgl.accessToken)

    try {
      if (map.current) return
      console.log('Initializing map...')

      // Initialize map
      map.current = new mapboxgl.Map({
        container: mapContainer.current,
        style: 'mapbox://styles/mapbox/dark-v11',
        center: [-122.4194, 37.7749], // San Francisco coordinates
        zoom: 12,
        preserveDrawingBuffer: true
      })
      console.log('Map initialized successfully')

      // Wait for map to load before adding controls
      map.current.on('load', () => {
        // Add navigation controls
        map.current.addControl(new mapboxgl.NavigationControl(), 'top-right')

        // Add geocoder (search box)
        const geocoder = new MapboxGeocoder({
          accessToken: mapboxgl.accessToken,
          mapboxgl: mapboxgl,
          marker: false,
          placeholder: 'Search for an address',
          bbox: [-122.5158, 37.7066, -122.3558, 37.8149], // Restrict to SF area
          flyTo: false // 禁用默认的 flyTo，使用我们自己的
        })

        map.current.addControl(geocoder)

        // Add marker
        marker.current = new mapboxgl.Marker({
          color: '#FFFFFF',
          scale: 1.2 // 稍微增大标记尺寸
        })

        // Handle location selection
        geocoder.on('result', (e) => {
          const coordinates = e.result.center
          
          // 先移除现有的标记（如果有）
          if (marker.current) {
            marker.current.remove()
          }
          
          // 添加新标记
          marker.current.setLngLat(coordinates).addTo(map.current)
          
          // 延迟一下再执行缩放，确保标记已经添加
          setTimeout(() => {
            flyToLocation(coordinates)
          }, 100)
          
          if (onLocationSelect) {
            onLocationSelect({
              lng: coordinates[0],
              lat: coordinates[1],
              address: e.result.place_name
            })
          }
        })

        // Handle click on map
        map.current.on('click', (e) => {
          const coordinates = [e.lngLat.lng, e.lngLat.lat]
          
          // 先移除现有的标记（如果有）
          if (marker.current) {
            marker.current.remove()
          }
          
          // 添加新标记
          marker.current.setLngLat(coordinates).addTo(map.current)
          
          // 延迟一下再执行缩放
          setTimeout(() => {
            flyToLocation(coordinates)
          }, 100)
          
          if (onLocationSelect) {
            onLocationSelect({
              lng: coordinates[0],
              lat: coordinates[1]
            })
          }
        })
      })

      // Cleanup function
      return () => {
        if (map.current) {
          map.current.remove()
          map.current = null
        }
      }
    } catch (error) {
      console.error('Error initializing map:', error)
    }
  }, [onLocationSelect])

  if (!mapboxgl.accessToken) {
    return (
      <div className="map-error">
        Map configuration error. Please check your settings.
      </div>
    )
  }

  return (
    <div 
      ref={mapContainer} 
      style={{ width: '100%', height: '100%', borderRadius: '4px' }}
    />
  )
}

MapPreview.propTypes = {
  onLocationSelect: PropTypes.func.isRequired
}

export default MapPreview 