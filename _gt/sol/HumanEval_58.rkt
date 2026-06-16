#lang racket
(define (common l1 l2)
  (sort (set->list (set-intersect (list->set l1) (list->set l2))) <))
