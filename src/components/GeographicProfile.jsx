import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import '../styles/GeographicProfile.css'
import MapPreview from './MapPreview'

function GeographicProfile() {
  const navigate = useNavigate()
  const [formData, setFormData] = useState({
    address: '',
    city: '',
    state: '',
    zipCode: '',
    residenceType: '',
    mobilityRange: '',
    coordinates: null
  })

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    })
  }

  const handleLocationSelect = (location) => {
    // Update form with selected location
    if (location.address) {
      const addressParts = location.address.split(',')
      setFormData(prev => ({
        ...prev,
        address: addressParts[0],
        city: addressParts[1]?.trim() || '',
        state: addressParts[2]?.trim() || '',
        coordinates: {
          lat: location.lat,
          lng: location.lng
        }
      }))
    } else {
      setFormData(prev => ({
        ...prev,
        coordinates: {
          lat: location.lat,
          lng: location.lng
        }
      }))
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    // TODO: Save data and proceed to next step
    navigate('/survey')
  }

  return (
    <div className="geographic-container">
      <div className="back-button" onClick={() => navigate('/')}>
        ← Back to Home
      </div>

      <div className="geographic-content">
        <h1>Define Your Geographic Profile</h1>
        
        <form onSubmit={handleSubmit} className="geographic-form">
          <div className="form-section map-section">
            <h2>Select Your Location</h2>
            <div className="map-container">
              <MapPreview onLocationSelect={handleLocationSelect} />
            </div>
          </div>

          <div className="form-section">
            <h2>Location Details</h2>
            
            <div className="input-group">
              <label>Street Address</label>
              <input
                type="text"
                name="address"
                value={formData.address}
                onChange={handleChange}
                placeholder="Enter your street address"
              />
            </div>

            <div className="input-row">
              <div className="input-group">
                <label>City</label>
                <input
                  type="text"
                  name="city"
                  value={formData.city}
                  onChange={handleChange}
                  placeholder="City"
                />
              </div>
              
              <div className="input-group">
                <label>State</label>
                <input
                  type="text"
                  name="state"
                  value={formData.state}
                  onChange={handleChange}
                  placeholder="State"
                />
              </div>
              
              <div className="input-group">
                <label>ZIP Code</label>
                <input
                  type="text"
                  name="zipCode"
                  value={formData.zipCode}
                  onChange={handleChange}
                  placeholder="ZIP Code"
                />
              </div>
            </div>
          </div>

          <div className="form-section">
            <h2>Residence Details</h2>
            
            <div className="input-group">
              <label>Type of Residence</label>
              <select 
                name="residenceType"
                value={formData.residenceType}
                onChange={handleChange}
              >
                <option value="">Select type...</option>
                <option value="apartment">Apartment</option>
                <option value="house">House</option>
                <option value="condo">Condominium</option>
                <option value="other">Other</option>
              </select>
            </div>

            <div className="input-group">
              <label>Daily Mobility Range</label>
              <select
                name="mobilityRange"
                value={formData.mobilityRange}
                onChange={handleChange}
              >
                <option value="">Select range...</option>
                <option value="local">Mostly within neighborhood</option>
                <option value="city">Throughout the city</option>
                <option value="regional">Regional (multiple cities)</option>
              </select>
            </div>
          </div>

          <div className="map-preview">
            {/* Map component will be added here */}
            <div className="map-placeholder">
              Map Preview
            </div>
          </div>

          <button type="submit" className="submit-button">
            Continue to Survey
          </button>
        </form>
      </div>
    </div>
  )
}

export default GeographicProfile 