class BaseAttack:
    """Base class for membership inference attacks"""
    
    def __init__(self, name: str):
        self.name = name
        self.attack_model = None
        self.metrics = {}
    
    def train(self, member_features, nonmember_features):
        """Train attack model"""
        raise NotImplementedError
    
    def predict(self, features) -> np.ndarray:
        """Predict membership"""
        raise NotImplementedError
    
    def evaluate(self, member_preds, nonmember_preds) -> Dict:
        """
        Evaluate attack performance
        
        Args:
            member_preds: Already computed predictions for members
            nonmember_preds: Already computed predictions for non-members
        """
        # Ensure predictions are 1D arrays
        if len(member_preds.shape) > 1:
            member_preds = member_preds.flatten()
        if len(nonmember_preds.shape) > 1:
            nonmember_preds = nonmember_preds.flatten()
        
        # Combine predictions and labels
        y_true = np.concatenate([
            np.ones(len(member_preds)),
            np.zeros(len(nonmember_preds))
        ])
        y_pred = np.concatenate([member_preds, nonmember_preds])
        
        # Calculate metrics
        auc = roc_auc_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred > 0.5)
        
        # Calculate precision and recall
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        
        self.metrics = {
            'auc': auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'member_predictions': member_preds,
            'nonmember_predictions': nonmember_preds
        }
        
        return self.metrics


class ConfidenceBasedAttack(BaseAttack):
    """
    Attack 1: Confidence-Based (Shokri et al., 2017)
    Uses prediction confidence scores to infer membership
    """
    
    def __init__(self):
        super().__init__("Confidence-Based (Shokri et al.)")
        self.attack_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    
    def _extract_features(self, confidences: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Extract features from model predictions
        
        Args:
            confidences: Prediction probabilities
            labels: True labels
            
        Returns:
            Feature matrix for attack model
        """
        features = []
        
        for conf, label in zip(confidences, labels):
            # Feature 1: Confidence for predicted class
            pred_conf = conf if label == 1 else 1 - conf
            
            # Feature 2: Confidence for true class
            true_conf = conf if label == 1 else 1 - conf
            
            # Feature 3: Entropy (uncertainty)
            epsilon = 1e-7
            entropy = -(conf * np.log(conf + epsilon) + 
                       (1-conf) * np.log(1-conf + epsilon))
            
            # Feature 4: Modified entropy
            modified_entropy = -np.log(pred_conf + epsilon)
            
            features.append([
                pred_conf,
                true_conf,
                entropy,
                modified_entropy
            ])
        
        return np.array(features)
    
    def train(
        self,
        member_confidences: np.ndarray,
        member_labels: np.ndarray,
        nonmember_confidences: np.ndarray,
        nonmember_labels: np.ndarray
    ):
        """Train attack model using shadow model approach"""
        
        # Extract features
        member_features = self._extract_features(member_confidences, member_labels)
        nonmember_features = self._extract_features(nonmember_confidences, nonmember_labels)
        
        # Create training data
        X_attack = np.vstack([member_features, nonmember_features])
        y_attack = np.concatenate([
            np.ones(len(member_features)),
            np.zeros(len(nonmember_features))
        ])
        
        # Train attack model
        self.attack_model.fit(X_attack, y_attack)
    
    def predict(self, confidences: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Predict membership probability"""
        features = self._extract_features(confidences, labels)
        return self.attack_model.predict_proba(features)[:, 1]


class LabelOnlyAttack(BaseAttack):
    """
    Attack 2: Label-Only (Choquette-Choo et al., 2021)
    Uses only predicted labels without confidence scores
    More realistic threat model
    """
    
    def __init__(self):
        super().__init__("Label-Only (Choquette-Choo et al.)")
        self.class_statistics = {}
    
    def train(
        self,
        member_predictions: np.ndarray,
        member_labels: np.ndarray,
        nonmember_predictions: np.ndarray,
        nonmember_labels: np.ndarray
    ):
        """
        Learn label statistics from shadow models
        """
        
        # Calculate agreement rates for members and non-members
        member_agreement = (member_predictions == member_labels).astype(float)
        nonmember_agreement = (nonmember_predictions == nonmember_labels).astype(float)
        
        # Per-class statistics
        for class_label in [0, 1]:
            member_mask = member_labels == class_label
            nonmember_mask = nonmember_labels == class_label
            
            self.class_statistics[class_label] = {
                'member_agreement': np.mean(member_agreement[member_mask]),
                'nonmember_agreement': np.mean(nonmember_agreement[nonmember_mask]),
                'member_count': np.sum(member_mask),
                'nonmember_count': np.sum(nonmember_mask)
            }
    
    def predict(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Predict membership based on label agreement
        """
        membership_scores = np.zeros(len(predictions))
        
        for i, (pred, label) in enumerate(zip(predictions, labels)):
            agrees = (pred == label)
            
            if label in self.class_statistics:
                stats = self.class_statistics[label]
                
                # Calculate likelihood ratio
                if agrees:
                    member_prob = stats['member_agreement']
                    nonmember_prob = stats['nonmember_agreement']
                else:
                    member_prob = 1 - stats['member_agreement']
                    nonmember_prob = 1 - stats['nonmember_agreement']
                
                # Avoid division by zero
                if nonmember_prob > 0:
                    likelihood_ratio = member_prob / nonmember_prob
                    membership_scores[i] = likelihood_ratio / (likelihood_ratio + 1)
                else:
                    membership_scores[i] = 0.5
            else:
                membership_scores[i] = 0.5
        
        return membership_scores
