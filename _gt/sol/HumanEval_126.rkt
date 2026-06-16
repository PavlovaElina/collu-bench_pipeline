#lang racket
(define (is_sorted lst)
  (cond [(< (length lst) 2) #t]
        [else (and (apply <= lst)
                   (for/and ([x (remove-duplicates lst)])
                     (<= (count (lambda (y) (= y x)) lst) 2)))]))
