#lang racket
(define (has_close_elements lst threshold)
  (for*/or ([i (in-range (length lst))]
            [j (in-range (length lst))])
    (and (not (= i j))
         (< (abs (- (list-ref lst i) (list-ref lst j))) threshold))))
